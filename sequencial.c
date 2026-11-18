// Filtro de Blur 3x3 Iterado - versão sequencial
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: gcc -O2 -o blur_seq sequencial.c  (ou make)
// Uso: ./blur_seq [-b] iteracoes pasta_saida frame0.pgm frame1.pgm ...
//   -b usa o filtro básico (9 leituras por pixel) em vez do otimizado

// Cada pixel interno recebe (soma dos 9 vizinhos + 4) / 9 e a borda da imagem
// não muda. Cada iteração lê de um buffer e escreve em outro.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <stdint.h>
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

#define MODO_BASICO 0
#define MODO_OTIMIZADO 1

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

    char assinatura[3];
    if (fscanf(arquivo, "%2s", assinatura) != 1 || strcmp(assinatura, "P5") != 0) {
        fprintf(stderr, "%s não é um PGM P5\n", caminho);
        exit(1);
    }

    int largura, altura, valor_maximo;
    int caractere;
    // Pula espaços e comentários antes das dimensões
    while (1) {
        caractere = fgetc(arquivo);
        if (caractere == EOF) {
            fprintf(stderr, "Cabeçalho inválido em %s\n", caminho);
            exit(1);
        }
        if (caractere == '#') {
            do {
                caractere = fgetc(arquivo);
            } while (caractere != '\n' && caractere != EOF);
        } else if (caractere != ' ' && caractere != '\t' && caractere != '\n' && caractere != '\r') {
            break;
        }
    }
    ungetc(caractere, arquivo);

    if (fscanf(arquivo, "%d %d %d", &largura, &altura, &valor_maximo) != 3) {
        fprintf(stderr, "Cabeçalho inválido em %s\n", caminho);
        exit(1);
    }
    fgetc(arquivo); // \n depois do cabeçalho

    if (largura <= 0 || altura <= 0 || valor_maximo != 255) {
        fprintf(stderr, "PGM não suportado: %s\n", caminho);
        exit(1);
    }

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

// Hash FNV-1a do resultado, para comparar com a versão paralela
uint32_t atualizar_checksum(uint32_t hash, const unsigned char *pixels, size_t total_pixels) {
    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        hash ^= pixels[posicao];
        hash *= 16777619u;
    }
    return hash;
}

// Uma iteração do blur somando os 9 vizinhos diretamente. A borda não é
// escrita porque os dois buffers já começam com ela.
void filtro_basico_uma_iteracao(const unsigned char *origem, unsigned char *destino, int largura, int altura) {
    if (altura < 3 || largura < 3) {
        return;
    }

    for (int linha = 1; linha < altura - 1; linha++) {
        for (int coluna = 1; coluna < largura - 1; coluna++) {
            int soma_vizinhanca = 0;
            for (int linha_vizinha = linha - 1; linha_vizinha <= linha + 1; linha_vizinha++) {
                for (int coluna_vizinha = coluna - 1; coluna_vizinha <= coluna + 1; coluna_vizinha++) {
                    soma_vizinhanca += origem[linha_vizinha * largura + coluna_vizinha];
                }
            }
            destino[linha * largura + coluna] = (unsigned char)((soma_vizinhanca + 4) / 9);
        }
    }
}

// Mesmo resultado do básico, mas reaproveita as somas das colunas para ler
// 3 pixels por ponto em vez de 9.
void filtro_otimizado_uma_iteracao(const unsigned char *origem, unsigned char *destino, int largura, int altura) {
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

// Executa as N iterações alternando entre os dois buffers. Recebe ponteiros
// para os buffers porque, ao final, o resultado precisa estar em *ponteiro_atual.
void executar_blur(
    unsigned char **ponteiro_atual,
    unsigned char **ponteiro_novo,
    int largura,
    int altura,
    int num_iteracoes,
    int modo) {
    unsigned char *buffer_atual = *ponteiro_atual;
    unsigned char *buffer_novo = *ponteiro_novo;

    for (int iteracao = 0; iteracao < num_iteracoes; iteracao++) {
        if (modo == MODO_OTIMIZADO) {
            filtro_otimizado_uma_iteracao(buffer_atual, buffer_novo, largura, altura);
        } else {
            filtro_basico_uma_iteracao(buffer_atual, buffer_novo, largura, altura);
        }

        unsigned char *ponteiro_temporario = buffer_atual;
        buffer_atual = buffer_novo;
        buffer_novo = ponteiro_temporario;
    }

    *ponteiro_atual = buffer_atual;
    *ponteiro_novo = buffer_novo;
}

typedef struct {
    unsigned char *buffer_atual;
    unsigned char *buffer_novo;
} AreaTrabalho;

// Copia o frame para os dois buffers (para a borda ficar certa nos dois) e
// aplica o blur. O resultado fica em area->buffer_atual.
void calcular_frame(Imagem *frame, AreaTrabalho *area, int num_iteracoes, int modo) {
    size_t total_pixels = (size_t)frame->largura * frame->altura;

    memcpy(area->buffer_atual, frame->pixels, total_pixels);
    memcpy(area->buffer_novo, frame->pixels, total_pixels);

    executar_blur(
        &area->buffer_atual,
        &area->buffer_novo,
        frame->largura,
        frame->altura,
        num_iteracoes,
        modo);
}

// Instante atual em segundos
double agora(void) {
    struct timespec instante;
    clock_gettime(CLOCK_MONOTONIC, &instante);
    return instante.tv_sec + instante.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int modo = MODO_OTIMIZADO;
    int opcao;
    int opcoes_validas = 1;
    while ((opcao = getopt(argc, argv, "b")) != -1) {
        if (opcao == 'b') {
            modo = MODO_BASICO;
        } else {
            opcoes_validas = 0;
        }
    }
    int num_frames = argc - optind - 2;
    if (!opcoes_validas || num_frames < 1) {
        fprintf(stderr, "Uso: %s [-b] iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        return 1;
    }

    int num_iteracoes = atoi(argv[optind]);
    const char *pasta_saida = argv[optind + 1];
    char **caminhos_frames = argv + optind + 2;

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

    AreaTrabalho area; // Reaproveitada em todos os frames
    area.buffer_atual = malloc(total_pixels);
    area.buffer_novo = malloc(total_pixels);
    if (!area.buffer_atual || !area.buffer_novo) {
        fprintf(stderr, "Erro de alocação\n");
        return 1;
    }

    uint32_t checksum = 2166136261u;

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
        calcular_frame(&frame, &area, num_iteracoes, modo);
        tempo_calculo += agora() - inicio;

        inicio = agora();
        char caminho[4096];
        snprintf(caminho, sizeof(caminho), "%s/blur_%04d.pgm", pasta_saida, indice);
        escrever_pgm(caminho, area.buffer_atual, largura, altura);
        tempo_escrita += agora() - inicio;

        checksum = atualizar_checksum(checksum, area.buffer_atual, total_pixels);
        free(frame.pixels);
    }

    double tempo_total = agora() - instante_inicio;
    printf("Frames: %d de %dx%d pixels\n", num_frames, largura, altura);
    printf("Vídeo processado em %.6f s (%d frames, modo %s, %d iterações, %.2f frames/s)\n",
        tempo_total, num_frames, modo == MODO_OTIMIZADO ? "otimizado" : "básico", num_iteracoes, num_frames / tempo_total);
    printf("Tempos: leitura %.3f s, cálculo %.3f s, escrita %.3f s\n",
        tempo_leitura, tempo_calculo, tempo_escrita);
    printf("Checksum do resultado: %08x\n", (unsigned)checksum);
    printf("Imagens salvas em %s/blur_*.pgm\n", pasta_saida);

    free(area.buffer_atual);
    free(area.buffer_novo);

    return 0;
}
