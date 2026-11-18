// Filtro de Blur 3x3 Iterado - versão paralela com MPI (OpenMPI)
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: mpicc -O2 -Wall -Wextra -o blur_par paralelo.c  (ou make)
// Execução:   mpirun -np P ./blur_par iteracoes pasta_saida frame0.pgm frame1.pgm ...

// A imagem é dividida em P faixas de linhas, uma por processo. O processo 0
// lê cada frame, distribui as faixas, calcula a sua e depois junta tudo e
// grava o resultado. Como o filtro 3x3 depende dos vizinhos, cada processo
// guarda uma cópia da última linha do vizinho de cima e da primeira linha do
// vizinho de baixo (halo). Antes de cada iteração os vizinhos trocam essas
// linhas. O resultado é idêntico ao do sequencial.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <mpi.h>
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

#define TAG_FAIXA 1
#define TAG_RESULTADO 2
#define TAG_HALO_PARA_CIMA 3
#define TAG_HALO_PARA_BAIXO 4

// Imagem em tons de cinza, pixel (l, c) em pixels[l * largura + c]
typedef struct {
    unsigned char *pixels;
    int largura;
    int altura;
} Imagem;

// Faixa de um processo. Os buffers têm num_linhas + 2 linhas: a linha 0 é o
// halo de cima, as linhas 1 a num_linhas são da faixa e a última é o halo de
// baixo. A cada iteração lê-se de um buffer e escreve-se no outro.
typedef struct {
    int linha_inicial; // Primeira linha da faixa na imagem
    int num_linhas;
    unsigned char *buffer_atual;
    unsigned char *buffer_novo;
} AreaTrabalho;

// Instante atual em segundos
double agora(void) {
    struct timespec instante;
    clock_gettime(CLOCK_MONOTONIC, &instante);
    return instante.tv_sec + instante.tv_nsec / 1e9;
}

// Lê um PGM binário (P5). Retorna 1 em sucesso e 0 em erro. Não encerra o
// programa em caso de erro para que todos os processos terminem juntos.
int ler_pgm(const char *caminho, Imagem *imagem) {
    FILE *arquivo = fopen(caminho, "rb");
    if (!arquivo) {
        fprintf(stderr, "Erro ao abrir %s\n", caminho);
        return 0;
    }

    int largura, altura, valor_maximo;
    if (fscanf(arquivo, "P5 %d %d %d", &largura, &altura, &valor_maximo) != 3 || valor_maximo != 255) {
        fprintf(stderr, "%s não é um PGM P5 de 8 bits\n", caminho);
        fclose(arquivo);
        return 0;
    }
    fgetc(arquivo); // \n depois do cabeçalho

    size_t total_pixels = (size_t)largura * altura;
    unsigned char *pixels = malloc(total_pixels);
    if (!pixels) {
        fprintf(stderr, "Erro de alocação ao ler %s\n", caminho);
        fclose(arquivo);
        return 0;
    }
    if (fread(pixels, 1, total_pixels, arquivo) != total_pixels) {
        fprintf(stderr, "Erro ao ler os pixels de %s\n", caminho);
        free(pixels);
        fclose(arquivo);
        return 0;
    }
    fclose(arquivo);

    imagem->pixels = pixels;
    imagem->largura = largura;
    imagem->altura = altura;
    return 1;
}

// Grava a imagem em PGM binário. Retorna 1 em sucesso.
int escrever_pgm(const char *caminho, const unsigned char *pixels, int largura, int altura) {
    FILE *arquivo = fopen(caminho, "wb");
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar %s\n", caminho);
        return 0;
    }
    size_t total_pixels = (size_t)largura * altura;
    fprintf(arquivo, "P5\n%d %d\n255\n", largura, altura);
    size_t gravados = fwrite(pixels, 1, total_pixels, arquivo);
    fclose(arquivo);
    if (gravados != total_pixels) {
        fprintf(stderr, "Erro ao gravar %s\n", caminho);
        return 0;
    }
    return 1;
}

// Faixa de linhas de um processo. As faixas diferem em no máximo 1 linha.
void calcular_faixa(int rank, int num_processos, int altura, int *linha_inicial, int *num_linhas) {
    *linha_inicial = rank * altura / num_processos;
    *num_linhas = (rank + 1) * altura / num_processos - *linha_inicial;
}

// Uma iteração do blur nas linhas locais [primeira_linha, ultima_linha].
// Reaproveita as somas das colunas para ler 3 pixels por ponto em vez de 9.
void filtro_linhas(const unsigned char *origem, unsigned char *destino, int largura, int primeira_linha, int ultima_linha) {
    for (int linha = primeira_linha; linha <= ultima_linha; linha++) {
        const unsigned char *linha_acima = origem + (size_t)(linha - 1) * largura;
        const unsigned char *linha_atual = origem + (size_t)linha * largura;
        const unsigned char *linha_abaixo = origem + (size_t)(linha + 1) * largura;
        unsigned char *linha_destino = destino + (size_t)linha * largura;

        // Bordas laterais não mudam
        linha_destino[0] = linha_atual[0];
        linha_destino[largura - 1] = linha_atual[largura - 1];
        if (largura < 3) {
            continue;
        }

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

// Troca uma linha com um vizinho. O menor rank do par envia antes de receber
// e o maior faz o contrário, para que os Send/Recv bloqueantes não travem.
void trocar_com_vizinho(int vizinho, unsigned char *linha_enviar, unsigned char *linha_receber, int largura,
                        int tag_enviar, int tag_receber, int sou_o_menor_rank) {
    if (sou_o_menor_rank) {
        MPI_Send(linha_enviar, largura, MPI_UNSIGNED_CHAR, vizinho, tag_enviar, MPI_COMM_WORLD);
        MPI_Recv(linha_receber, largura, MPI_UNSIGNED_CHAR, vizinho, tag_receber, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(linha_receber, largura, MPI_UNSIGNED_CHAR, vizinho, tag_receber, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(linha_enviar, largura, MPI_UNSIGNED_CHAR, vizinho, tag_enviar, MPI_COMM_WORLD);
    }
}

// Envia a primeira linha da faixa para o vizinho de cima e a última para o de
// baixo, recebendo deles as linhas de halo. Ranks pares falam primeiro com o
// vizinho de baixo e ímpares com o de cima, assim cada par se encontra ao
// mesmo tempo. Na troca com o vizinho de baixo eu sou o menor rank do par.
void trocar_halos(const AreaTrabalho *area, unsigned char *buffer, int largura, int rank, int num_processos) {
    unsigned char *halo_de_cima = buffer;
    unsigned char *primeira_linha = buffer + (size_t)largura;
    unsigned char *ultima_linha = buffer + (size_t)area->num_linhas * largura;
    unsigned char *halo_de_baixo = buffer + (size_t)(area->num_linhas + 1) * largura;

    int tem_cima = (rank > 0);
    int tem_baixo = (rank < num_processos - 1);

    if (rank % 2 == 0) {
        if (tem_baixo) trocar_com_vizinho(rank + 1, ultima_linha, halo_de_baixo, largura, TAG_HALO_PARA_BAIXO, TAG_HALO_PARA_CIMA, 1);
        if (tem_cima) trocar_com_vizinho(rank - 1, primeira_linha, halo_de_cima, largura, TAG_HALO_PARA_CIMA, TAG_HALO_PARA_BAIXO, 0);
    } else {
        if (tem_cima) trocar_com_vizinho(rank - 1, primeira_linha, halo_de_cima, largura, TAG_HALO_PARA_CIMA, TAG_HALO_PARA_BAIXO, 0);
        if (tem_baixo) trocar_com_vizinho(rank + 1, ultima_linha, halo_de_baixo, largura, TAG_HALO_PARA_BAIXO, TAG_HALO_PARA_CIMA, 1);
    }
}

// Executa as N iterações. Antes de cada uma os processos trocam os halos.
// A linha 0 e a última linha da imagem não mudam, então não são calculadas.
void executar_blur(AreaTrabalho *area, int largura, int altura, int rank, int num_processos, int num_iteracoes,
                   double *tempo_calculo, double *tempo_comunicacao) {
    unsigned char *buffer_atual = area->buffer_atual;
    unsigned char *buffer_novo = area->buffer_novo;

    int primeira = (area->linha_inicial == 0) ? 2 : 1;
    int ultima = (area->linha_inicial + area->num_linhas == altura) ? area->num_linhas - 1 : area->num_linhas;

    for (int iteracao = 0; iteracao < num_iteracoes; iteracao++) {
        double inicio = agora();
        trocar_halos(area, buffer_atual, largura, rank, num_processos);
        *tempo_comunicacao += agora() - inicio;

        inicio = agora();
        filtro_linhas(buffer_atual, buffer_novo, largura, primeira, ultima);
        *tempo_calculo += agora() - inicio;

        unsigned char *ponteiro_temporario = buffer_atual;
        buffer_atual = buffer_novo;
        buffer_novo = ponteiro_temporario;
    }

    area->buffer_atual = buffer_atual;
    area->buffer_novo = buffer_novo;
}

// O processo 0 envia a cada processo a sua faixa do frame, que fica a partir
// da linha 1 do buffer. Os dois buffers ficam iguais para que as linhas que
// não são calculadas existam nos dois.
void distribuir_faixas(const Imagem *frame, AreaTrabalho *area, int largura, int altura, int rank, int num_processos) {
    if (rank == 0) {
        for (int destino = 1; destino < num_processos; destino++) {
            int linha_inicial_destino, linhas_destino;
            calcular_faixa(destino, num_processos, altura, &linha_inicial_destino, &linhas_destino);
            MPI_Send(frame->pixels + (size_t)linha_inicial_destino * largura, linhas_destino * largura,
                     MPI_UNSIGNED_CHAR, destino, TAG_FAIXA, MPI_COMM_WORLD);
        }
        memcpy(area->buffer_atual + largura, frame->pixels + (size_t)area->linha_inicial * largura,
               (size_t)area->num_linhas * largura);
    } else {
        MPI_Recv(area->buffer_atual + largura, area->num_linhas * largura, MPI_UNSIGNED_CHAR, 0,
                 TAG_FAIXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    memcpy(area->buffer_novo, area->buffer_atual, (size_t)(area->num_linhas + 2) * largura);
}

// O processo 0 junta as faixas calculadas na imagem final
void coletar_faixas(const AreaTrabalho *area, unsigned char *resultado, int largura, int altura, int rank, int num_processos) {
    if (rank == 0) {
        for (int origem = 1; origem < num_processos; origem++) {
            int linha_inicial_origem, linhas_origem;
            calcular_faixa(origem, num_processos, altura, &linha_inicial_origem, &linhas_origem);
            MPI_Recv(resultado + (size_t)linha_inicial_origem * largura, linhas_origem * largura, MPI_UNSIGNED_CHAR,
                     origem, TAG_RESULTADO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        memcpy(resultado + (size_t)area->linha_inicial * largura, area->buffer_atual + largura,
               (size_t)area->num_linhas * largura);
    } else {
        MPI_Send(area->buffer_atual + largura, area->num_linhas * largura, MPI_UNSIGNED_CHAR, 0,
                 TAG_RESULTADO, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int num_processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    int num_frames = argc - 3;
    if (num_frames < 1) {
        if (rank == 0) {
            fprintf(stderr, "Uso: mpirun -np P %s iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int num_iteracoes = atoi(argv[1]);
    const char *pasta_saida = argv[2];
    char **caminhos_frames = argv + 3;
    if (num_iteracoes <= 0) {
        if (rank == 0) {
            fprintf(stderr, "iteracoes precisa ser um inteiro positivo\n");
        }
        MPI_Finalize();
        return 1;
    }

    // O processo 0 lê o primeiro frame e repassa as dimensões
    Imagem frame = {NULL, 0, 0};
    int dimensoes[3] = {0, 0, 0}; // leitura ok, largura, altura
    double tempo_leitura = 0.0;
    if (rank == 0) {
        if (CRIAR_PASTA(pasta_saida) != 0 && errno != EEXIST) {
            fprintf(stderr, "Erro ao criar a pasta %s\n", pasta_saida);
        } else {
            double inicio = agora();
            if (ler_pgm(caminhos_frames[0], &frame)) {
                dimensoes[0] = 1;
                dimensoes[1] = frame.largura;
                dimensoes[2] = frame.altura;
            }
            tempo_leitura += agora() - inicio;
        }
    }
    MPI_Bcast(dimensoes, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (!dimensoes[0]) {
        MPI_Finalize();
        return 1;
    }
    int largura = dimensoes[1];
    int altura = dimensoes[2];
    size_t total_pixels = (size_t)largura * altura;

    if (altura < num_processos) {
        if (rank == 0) {
            fprintf(stderr, "Mais processos (%d) do que linhas na imagem (%d)\n", num_processos, altura);
        }
        free(frame.pixels);
        MPI_Finalize();
        return 1;
    }
    // Buffers reaproveitados em todos os frames
    AreaTrabalho area;
    calcular_faixa(rank, num_processos, altura, &area.linha_inicial, &area.num_linhas);
    area.buffer_atual = calloc((size_t)(area.num_linhas + 2) * largura, 1);
    area.buffer_novo = calloc((size_t)(area.num_linhas + 2) * largura, 1);
    unsigned char *imagem_resultado = (rank == 0) ? malloc(total_pixels) : NULL;

    double tempo_distribuicao = 0.0;
    double tempo_calculo = 0.0;
    double tempo_comunicacao = 0.0;
    double tempo_coleta = 0.0;
    double tempo_escrita = 0.0;
    int execucao_com_sucesso = 1;
    int frames_processados = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double instante_inicio = agora();

    for (int indice = 0; indice < num_frames; indice++) {
        // O processo 0 lê o frame e avisa os demais se deu certo
        int continuar = 1;
        if (rank == 0 && indice > 0) {
            double inicio = agora();
            free(frame.pixels);
            frame.pixels = NULL;
            if (!ler_pgm(caminhos_frames[indice], &frame)) {
                continuar = 0;
            } else if (frame.largura != largura || frame.altura != altura) {
                fprintf(stderr, "%s tem dimensões diferentes do primeiro frame\n", caminhos_frames[indice]);
                continuar = 0;
            }
            tempo_leitura += agora() - inicio;
        }
        MPI_Bcast(&continuar, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!continuar) {
            execucao_com_sucesso = 0;
            break;
        }

        double inicio = agora();
        distribuir_faixas(&frame, &area, largura, altura, rank, num_processos);
        tempo_distribuicao += agora() - inicio;

        executar_blur(&area, largura, altura, rank, num_processos, num_iteracoes, &tempo_calculo, &tempo_comunicacao);

        inicio = agora();
        coletar_faixas(&area, imagem_resultado, largura, altura, rank, num_processos);
        tempo_coleta += agora() - inicio;

        if (rank == 0) {
            inicio = agora();
            char caminho[4096];
            snprintf(caminho, sizeof(caminho), "%s/blur_%04d.pgm", pasta_saida, indice);
            if (!escrever_pgm(caminho, imagem_resultado, largura, altura)) {
                execucao_com_sucesso = 0;
            }
            tempo_escrita += agora() - inicio;
        }
        frames_processados++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tempo_total = agora() - instante_inicio;

    if (rank == 0) {
        printf("Frames: %d de %dx%d pixels\n", frames_processados, largura, altura);
        printf("Vídeo processado em %.6f s (%d frames, %d processos, %d iterações, %.2f frames/s)\n",
               tempo_total, frames_processados, num_processos, num_iteracoes, frames_processados / tempo_total);
        printf("Tempos do processo 0: leitura %.3f s, distribuição %.3f s, cálculo %.3f s, halos %.3f s, coleta %.3f s, escrita %.3f s\n",
               tempo_leitura, tempo_distribuicao, tempo_calculo, tempo_comunicacao, tempo_coleta, tempo_escrita);
        printf("Imagens salvas em %s/blur_*.pgm\n", pasta_saida);
    }

    free(area.buffer_atual);
    free(area.buffer_novo);
    free(imagem_resultado);
    free(frame.pixels);
    MPI_Finalize();
    return execucao_com_sucesso ? 0 : 1;
}
