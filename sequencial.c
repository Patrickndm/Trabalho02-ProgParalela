// Filtro de Blur 3x3 Iterado - versão sequencial
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: gcc -O2 -o blur_seq sequencial.c  (ou make)
// Uso: ./blur_seq iteracoes pasta_saida frame0.pgm frame1.pgm ...

// Cada pixel interno recebe (soma dos 9 vizinhos + 4) / 9 e a borda da imagem
// não muda. Cada iteração lê de um buffer e escreve em outro.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

// No Windows (MinGW) o mkdir recebe só 1 argumento
#ifdef _WIN32
#include <direct.h>
#define CRIAR_PASTA(caminho) _mkdir(caminho)
#else
#define CRIAR_PASTA(caminho) mkdir(caminho, 0755)
#endif

// Imagem em tons de cinza, pixel (l, c) em pixels[l * largura + c]
typedef struct {
    unsigned char *pixels;
    int largura;
    int altura;
} Imagem;

// Lê um PGM binário (P5). Encerra o programa em caso de erro.
Imagem ler_pgm(const char *caminho) {
    FILE *arquivo = fopen(caminho, "rb");
    if (!arquivo) {
        fprintf(stderr, "Erro ao abrir %s\n", caminho);
        exit(1);
    }

    int largura, altura, valor_maximo;
    if (fscanf(arquivo, "P5 %d %d %d", &largura, &altura, &valor_maximo) != 3 || valor_maximo != 255) {
        fprintf(stderr, "%s não é um PGM P5 de 8 bits\n", caminho);
        exit(1);
    }
    fgetc(arquivo); // \n depois do cabeçalho

    unsigned char *pixels = malloc((size_t)largura * altura);
    if (!pixels) {
        fprintf(stderr, "Erro de alocação ao ler %s\n", caminho);
        exit(1);
    }
    if (fread(pixels, 1, (size_t)largura * altura, arquivo) != (size_t)largura * altura) {
        fprintf(stderr, "Erro ao ler os pixels de %s\n", caminho);
        exit(1);
    }
    fclose(arquivo);

    Imagem imagem = {pixels, largura, altura};
    return imagem;
}

void escrever_pgm(const char *caminho, const unsigned char *pixels, int largura, int altura) {
    FILE *arquivo = fopen(caminho, "wb");
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar %s\n", caminho);
        exit(1);
    }
    fprintf(arquivo, "P5\n%d %d\n255\n", largura, altura);
    fwrite(pixels, 1, (size_t)largura * altura, arquivo);
    fclose(arquivo);
}

// Uma iteração do blur. Reaproveita as somas das colunas para ler 3 pixels
// por ponto em vez de 9. A borda não é escrita porque os dois buffers já
// começam com ela.
void filtro_uma_iteracao(const unsigned char *origem, unsigned char *destino, int largura, int altura) {
    if (altura < 3 || largura < 3) {
        return;
    }

    for (int linha = 1; linha < altura - 1; linha++) {
        const unsigned char *linha_acima = origem + (size_t)(linha - 1) * largura;
        const unsigned char *linha_atual = origem + (size_t)linha * largura;
        const unsigned char *linha_abaixo = origem + (size_t)(linha + 1) * largura;
        unsigned char *linha_destino = destino + (size_t)linha * largura;

        int soma_coluna_anterior = linha_acima[0] + linha_atual[0] + linha_abaixo[0];
        int soma_coluna_central = linha_acima[1] + linha_atual[1] + linha_abaixo[1];

        for (int coluna = 1; coluna < largura - 1; coluna++) {
            int soma_coluna_seguinte = linha_acima[coluna + 1] + linha_atual[coluna + 1] + linha_abaixo[coluna + 1];
            int soma_vizinhanca = soma_coluna_anterior + soma_coluna_central + soma_coluna_seguinte;
            linha_destino[coluna] = (unsigned char)((soma_vizinhanca + 4) / 9);

            soma_coluna_anterior = soma_coluna_central;
            soma_coluna_central = soma_coluna_seguinte;
        }
    }
}

// Executa as N iterações alternando entre os dois buffers, que precisam
// começar iguais. Retorna o buffer que ficou com o resultado.
unsigned char *executar_blur(unsigned char *buffer_atual, unsigned char *buffer_novo, int largura, int altura, int num_iteracoes) {
    for (int iteracao = 0; iteracao < num_iteracoes; iteracao++) {
        filtro_uma_iteracao(buffer_atual, buffer_novo, largura, altura);

        unsigned char *ponteiro_temporario = buffer_atual;
        buffer_atual = buffer_novo;
        buffer_novo = ponteiro_temporario;
    }
    return buffer_atual;
}

// Instante atual em segundos
double agora(void) {
    struct timespec instante;
    clock_gettime(CLOCK_MONOTONIC, &instante);
    return instante.tv_sec + instante.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int num_frames = argc - 3;
    if (num_frames < 1) {
        fprintf(stderr, "Uso: %s iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        return 1;
    }

    int num_iteracoes = atoi(argv[1]);
    const char *pasta_saida = argv[2];
    char **caminhos_frames = argv + 3;

    if (num_iteracoes <= 0) {
        fprintf(stderr, "iteracoes precisa ser um inteiro positivo\n");
        return 1;
    }
    if (CRIAR_PASTA(pasta_saida) != 0 && errno != EEXIST) {
        fprintf(stderr, "Erro ao criar a pasta %s\n", pasta_saida);
        return 1;
    }

    double tempo_leitura = 0.0;
    double tempo_calculo = 0.0;
    double tempo_escrita = 0.0;
    double instante_inicio = agora();

    // O primeiro frame é lido antes do laço para saber as dimensões
    double inicio = agora();
    Imagem frame = ler_pgm(caminhos_frames[0]);
    tempo_leitura += agora() - inicio;
    int largura = frame.largura;
    int altura = frame.altura;
    size_t total_pixels = (size_t)largura * altura;

    // Buffers reaproveitados em todos os frames
    unsigned char *buffer_a = malloc(total_pixels);
    unsigned char *buffer_b = malloc(total_pixels);

    for (int indice = 0; indice < num_frames; indice++) {
        if (indice > 0) {
            inicio = agora();
            frame = ler_pgm(caminhos_frames[indice]);
            tempo_leitura += agora() - inicio;
            if (frame.largura != largura || frame.altura != altura) {
                fprintf(stderr, "%s tem dimensões diferentes do primeiro frame\n", caminhos_frames[indice]);
                return 1;
            }
        }

        inicio = agora();
        memcpy(buffer_a, frame.pixels, total_pixels);
        memcpy(buffer_b, frame.pixels, total_pixels);
        unsigned char *resultado = executar_blur(buffer_a, buffer_b, largura, altura, num_iteracoes);
        tempo_calculo += agora() - inicio;

        inicio = agora();
        char caminho[4096];
        snprintf(caminho, sizeof(caminho), "%s/blur_%04d.pgm", pasta_saida, indice);
        escrever_pgm(caminho, resultado, largura, altura);
        tempo_escrita += agora() - inicio;

        free(frame.pixels);
    }

    double tempo_total = agora() - instante_inicio;
    printf("Frames: %d de %dx%d pixels\n", num_frames, largura, altura);
    printf("Vídeo processado em %.6f s (%d frames, %d iterações, %.2f frames/s)\n",
        tempo_total, num_frames, num_iteracoes, num_frames / tempo_total);
    printf("Tempos: leitura %.3f s, cálculo %.3f s, escrita %.3f s\n",
        tempo_leitura, tempo_calculo, tempo_escrita);
    printf("Imagens salvas em %s/blur_*.pgm\n", pasta_saida);

    free(buffer_a);
    free(buffer_b);

    return 0;
}
