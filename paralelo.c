// Filtro de Blur 3x3 Iterado - versão paralela com MPI (OpenMPI)
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: mpicc -O2 -Wall -Wextra -o blur_par paralelo.c  (ou make)
// Execução:   mpirun -np P ./blur_par [-b bloco] [-v pasta_referencia]
//                 iteracoes pasta_saida frame0.pgm frame1.pgm ...

// A imagem é dividida em P faixas de linhas, uma por processo. O processo 0
// lê cada frame, distribui as faixas, calcula a sua e depois junta tudo e
// grava o resultado. Como o filtro 3x3 depende dos vizinhos, cada processo
// guarda também algumas linhas dos vizinhos de cima e de baixo (halo).
//
// O halo tem até B linhas (opção -b). Os processos trocam halos uma vez e
// fazem B iterações seguidas sem se comunicar, calculando uma linha a menos
// em cada ponta a cada iteração. Ao final das B iterações as linhas da faixa
// estão corretas. Com B = 1 a troca acontece em toda iteração.
//
// O resultado é idêntico ao do sequencial, o que pode ser conferido com -v.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <mpi.h>
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

#define BLOCO_PADRAO 4 // Iterações por rodada quando -b não é informado

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

// Faixa de um processo com os halos. Os buffers guardam, em ordem, o halo de
// cima, as linhas da faixa e o halo de baixo. A cada iteração lê-se de um e
// escreve-se no outro.
typedef struct {
    int linha_inicial; // Primeira linha da faixa na imagem
    int num_linhas;
    int halo_cima;
    int halo_baixo;
    int linhas_enviar_cima;
    int linhas_enviar_baixo;
    int fixa_em_cima; // 1 se a primeira linha local é a linha 0 da imagem
    int fixa_em_baixo; // 1 se a última linha local é a última da imagem
    int linhas_locais; // halo_cima + num_linhas + halo_baixo
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

    char assinatura[3];
    if (fscanf(arquivo, "%2s", assinatura) != 1 || strcmp(assinatura, "P5") != 0) {
        fprintf(stderr, "%s não é um PGM P5\n", caminho);
        fclose(arquivo);
        return 0;
    }

    int largura, altura, valor_maximo;
    int caractere;
    // Pula espaços e comentários antes das dimensões
    while (1) {
        caractere = fgetc(arquivo);
        if (caractere == EOF) {
            fprintf(stderr, "Cabeçalho inválido em %s\n", caminho);
            fclose(arquivo);
            return 0;
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
        fclose(arquivo);
        return 0;
    }
    fgetc(arquivo); // \n depois do cabeçalho

    if (largura <= 0 || altura <= 0 || valor_maximo != 255) {
        fprintf(stderr, "PGM não suportado: %s\n", caminho);
        fclose(arquivo);
        return 0;
    }

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

// Hash FNV-1a do resultado, o mesmo do sequencial, para comparar as versões
uint32_t atualizar_checksum(uint32_t hash, const unsigned char *pixels, size_t total_pixels) {
    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        hash ^= pixels[posicao];
        hash *= 16777619u;
    }
    return hash;
}

// Calcula a faixa de linhas de um processo. Os primeiros altura % num_processos
// processos ficam com uma linha a mais.
void calcular_faixa(int rank, int num_processos, int altura, int *linha_inicial, int *num_linhas) {
    int linhas_base = altura / num_processos;
    int linhas_extras = altura % num_processos;
    if (rank < linhas_extras) {
        *num_linhas = linhas_base + 1;
        *linha_inicial = rank * (linhas_base + 1);
    } else {
        *num_linhas = linhas_base;
        *linha_inicial = linhas_extras * (linhas_base + 1) + (rank - linhas_extras) * linhas_base;
    }
}

// Monta a faixa, os halos e os buffers de um processo. Cada halo tem no
// máximo 'bloco' linhas. A main garante que bloco <= menor faixa, então o
// vizinho sempre tem linhas suficientes para enviar.
int preparar_area(AreaTrabalho *area, int rank, int num_processos, int largura, int altura, int bloco) {
    calcular_faixa(rank, num_processos, altura, &area->linha_inicial, &area->num_linhas);
    int linha_depois_da_faixa = area->linha_inicial + area->num_linhas;
    int linhas_acima = area->linha_inicial;
    int linhas_abaixo = altura - linha_depois_da_faixa;

    area->halo_cima = (linhas_acima < bloco) ? linhas_acima : bloco;
    area->halo_baixo = (linhas_abaixo < bloco) ? linhas_abaixo : bloco;
    // Envio a cada vizinho o tamanho do halo dele, calculado pela mesma fórmula
    area->linhas_enviar_cima = (rank > 0) ? ((altura - area->linha_inicial < bloco) ? altura - area->linha_inicial : bloco) : 0;
    area->linhas_enviar_baixo = (rank < num_processos - 1) ? ((linha_depois_da_faixa < bloco) ? linha_depois_da_faixa : bloco) : 0;

    area->fixa_em_cima = (area->linha_inicial - area->halo_cima == 0);
    area->fixa_em_baixo = (linha_depois_da_faixa + area->halo_baixo == altura);
    area->linhas_locais = area->halo_cima + area->num_linhas + area->halo_baixo;

    size_t tamanho_buffer = (size_t)area->linhas_locais * largura;
    area->buffer_atual = calloc(tamanho_buffer, 1);
    area->buffer_novo = calloc(tamanho_buffer, 1);
    if (!area->buffer_atual || !area->buffer_novo) {
        free(area->buffer_atual);
        free(area->buffer_novo);
        area->buffer_atual = NULL;
        area->buffer_novo = NULL;
        return 0;
    }
    return 1;
}

void liberar_area(AreaTrabalho *area) {
    free(area->buffer_atual);
    free(area->buffer_novo);
    area->buffer_atual = NULL;
    area->buffer_novo = NULL;
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

// Troca linhas com um vizinho. O menor rank do par envia antes de receber e o
// maior faz o contrário, para que os Send/Recv bloqueantes não travem.
void trocar_com_vizinho(int vizinho, const unsigned char *dados_enviar, int bytes_enviar,
                        unsigned char *dados_receber, int bytes_receber,
                        int tag_enviar, int tag_receber, int sou_o_menor_rank) {
    if (sou_o_menor_rank) {
        if (bytes_enviar > 0) {
            MPI_Send((void *)dados_enviar, bytes_enviar, MPI_UNSIGNED_CHAR, vizinho, tag_enviar, MPI_COMM_WORLD);
        }
        if (bytes_receber > 0) {
            MPI_Recv(dados_receber, bytes_receber, MPI_UNSIGNED_CHAR, vizinho, tag_receber, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        if (bytes_receber > 0) {
            MPI_Recv(dados_receber, bytes_receber, MPI_UNSIGNED_CHAR, vizinho, tag_receber, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (bytes_enviar > 0) {
            MPI_Send((void *)dados_enviar, bytes_enviar, MPI_UNSIGNED_CHAR, vizinho, tag_enviar, MPI_COMM_WORLD);
        }
    }
}

// Atualiza os halos com as linhas dos vizinhos e envia a eles as minhas.
// Ranks pares falam primeiro com o vizinho de baixo e ímpares com o de cima,
// assim cada par se encontra ao mesmo tempo.
void trocar_halos(const AreaTrabalho *area, unsigned char *buffer, int largura, int rank, int num_processos) {
    unsigned char *halo_de_cima = buffer;
    unsigned char *primeiras_proprias = buffer + (size_t)area->halo_cima * largura;
    unsigned char *ultimas_proprias = buffer + (size_t)(area->halo_cima + area->num_linhas - area->linhas_enviar_baixo) * largura;
    unsigned char *halo_de_baixo = buffer + (size_t)(area->halo_cima + area->num_linhas) * largura;

    for (int etapa = 0; etapa < 2; etapa++) {
        int com_baixo = (etapa == 0) == (rank % 2 == 0);
        if (com_baixo && rank < num_processos - 1) {
            trocar_com_vizinho(rank + 1,
                               ultimas_proprias, area->linhas_enviar_baixo * largura,
                               halo_de_baixo, area->halo_baixo * largura,
                               TAG_HALO_PARA_BAIXO, TAG_HALO_PARA_CIMA, 1);
        } else if (!com_baixo && rank > 0) {
            trocar_com_vizinho(rank - 1,
                               primeiras_proprias, area->linhas_enviar_cima * largura,
                               halo_de_cima, area->halo_cima * largura,
                               TAG_HALO_PARA_CIMA, TAG_HALO_PARA_BAIXO, 0);
        }
    }
}

// Executa as N iterações em rodadas. Cada rodada troca os halos uma vez e faz
// até 'bloco' iterações. Na iteração m da rodada as m linhas de cada ponta já
// estão desatualizadas, então o cálculo vai da linha m até linhas_locais-1-m
// (exceto nas pontas que são borda da imagem).
void executar_blur(AreaTrabalho *area, int largura, int rank, int num_processos, int num_iteracoes, int bloco,
                   double *tempo_calculo, double *tempo_comunicacao) {
    unsigned char *buffer_atual = area->buffer_atual;
    unsigned char *buffer_novo = area->buffer_novo;
    int iteracoes_restantes = num_iteracoes;

    while (iteracoes_restantes > 0) {
        int iteracoes_da_rodada = (iteracoes_restantes < bloco) ? iteracoes_restantes : bloco;

        double inicio = agora();
        trocar_halos(area, buffer_atual, largura, rank, num_processos);
        // O outro buffer também precisa dos halos recebidos
        memcpy(buffer_novo, buffer_atual, (size_t)area->halo_cima * largura);
        size_t inicio_halo_baixo = (size_t)(area->halo_cima + area->num_linhas) * largura;
        memcpy(buffer_novo + inicio_halo_baixo, buffer_atual + inicio_halo_baixo, (size_t)area->halo_baixo * largura);
        *tempo_comunicacao += agora() - inicio;

        inicio = agora();
        for (int passo = 1; passo <= iteracoes_da_rodada; passo++) {
            int primeira_linha = area->fixa_em_cima ? 1 : passo;
            int ultima_linha = area->fixa_em_baixo ? area->linhas_locais - 2 : area->linhas_locais - 1 - passo;
            filtro_linhas(buffer_atual, buffer_novo, largura, primeira_linha, ultima_linha);

            unsigned char *ponteiro_temporario = buffer_atual;
            buffer_atual = buffer_novo;
            buffer_novo = ponteiro_temporario;
        }
        *tempo_calculo += agora() - inicio;

        iteracoes_restantes -= iteracoes_da_rodada;
    }

    area->buffer_atual = buffer_atual;
    area->buffer_novo = buffer_novo;
}

// O processo 0 envia a cada processo a sua faixa do frame
void distribuir_faixas(const Imagem *frame, AreaTrabalho *area, int largura, int altura, int rank, int num_processos) {
    size_t deslocamento_faixa = (size_t)area->halo_cima * largura;
    if (rank == 0) {
        for (int destino = 1; destino < num_processos; destino++) {
            int linha_inicial_destino, linhas_destino;
            calcular_faixa(destino, num_processos, altura, &linha_inicial_destino, &linhas_destino);
            MPI_Send(frame->pixels + (size_t)linha_inicial_destino * largura, linhas_destino * largura,
                     MPI_UNSIGNED_CHAR, destino, TAG_FAIXA, MPI_COMM_WORLD);
        }
        memcpy(area->buffer_atual + deslocamento_faixa, frame->pixels + (size_t)area->linha_inicial * largura,
               (size_t)area->num_linhas * largura);
    } else {
        MPI_Recv(area->buffer_atual + deslocamento_faixa, area->num_linhas * largura, MPI_UNSIGNED_CHAR, 0,
                 TAG_FAIXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    memcpy(area->buffer_novo, area->buffer_atual, (size_t)area->linhas_locais * largura);
}

// O processo 0 junta as faixas calculadas na imagem final
void coletar_faixas(const AreaTrabalho *area, unsigned char *resultado, int largura, int altura, int rank, int num_processos) {
    size_t deslocamento_faixa = (size_t)area->halo_cima * largura;
    if (rank == 0) {
        for (int origem = 1; origem < num_processos; origem++) {
            int linha_inicial_origem, linhas_origem;
            calcular_faixa(origem, num_processos, altura, &linha_inicial_origem, &linhas_origem);
            MPI_Recv(resultado + (size_t)linha_inicial_origem * largura, linhas_origem * largura, MPI_UNSIGNED_CHAR,
                     origem, TAG_RESULTADO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        memcpy(resultado + (size_t)area->linha_inicial * largura, area->buffer_atual + deslocamento_faixa,
               (size_t)area->num_linhas * largura);
    } else {
        MPI_Send(area->buffer_atual + deslocamento_faixa, area->num_linhas * largura, MPI_UNSIGNED_CHAR, 0,
                 TAG_RESULTADO, MPI_COMM_WORLD);
    }
}

// Usada com -v. Compara pixel a pixel a saída do paralelo com a do sequencial.
// Retorna 1 se todos os frames forem idênticos.
int verificar_resultados(const char *pasta_saida, const char *pasta_referencia, int num_frames) {
    int tudo_igual = 1;
    for (int indice = 0; indice < num_frames; indice++) {
        char caminho_paralelo[4096];
        char caminho_referencia[4096];
        snprintf(caminho_paralelo, sizeof(caminho_paralelo), "%s/blur_%04d.pgm", pasta_saida, indice);
        snprintf(caminho_referencia, sizeof(caminho_referencia), "%s/blur_%04d.pgm", pasta_referencia, indice);

        Imagem imagem_paralela, imagem_referencia;
        if (!ler_pgm(caminho_paralelo, &imagem_paralela)) {
            tudo_igual = 0;
            continue;
        }
        if (!ler_pgm(caminho_referencia, &imagem_referencia)) {
            free(imagem_paralela.pixels);
            tudo_igual = 0;
            continue;
        }

        if (imagem_paralela.largura != imagem_referencia.largura || imagem_paralela.altura != imagem_referencia.altura) {
            printf("Frame %d: dimensões diferentes\n", indice);
            tudo_igual = 0;
        } else {
            size_t total_pixels = (size_t)imagem_paralela.largura * imagem_paralela.altura;
            size_t diferencas = 0;
            for (size_t posicao = 0; posicao < total_pixels; posicao++) {
                if (imagem_paralela.pixels[posicao] != imagem_referencia.pixels[posicao]) {
                    diferencas++;
                }
            }
            if (diferencas == 0) {
                printf("Frame %d: igual ao sequencial\n", indice);
            } else {
                printf("Frame %d: %zu pixels diferentes\n", indice, diferencas);
                tudo_igual = 0;
            }
        }
        free(imagem_paralela.pixels);
        free(imagem_referencia.pixels);
    }
    printf(tudo_igual ? "Verificação OK\n" : "Verificação falhou\n");
    return tudo_igual;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int num_processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    int bloco = BLOCO_PADRAO;
    const char *pasta_referencia = NULL;
    int opcao;
    int opcoes_validas = 1;
    if (rank != 0) {
        opterr = 0; // Só o processo 0 imprime os erros do getopt
    }
    while ((opcao = getopt(argc, argv, "b:v:")) != -1) {
        if (opcao == 'b') {
            bloco = atoi(optarg);
        } else if (opcao == 'v') {
            pasta_referencia = optarg;
        } else {
            opcoes_validas = 0;
        }
    }
    int num_frames = argc - optind - 2;
    if (!opcoes_validas || num_frames < 1 || bloco < 1) {
        if (rank == 0) {
            fprintf(stderr, "Uso: mpirun -np P %s [-b bloco] [-v pasta_referencia] iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int num_iteracoes = atoi(argv[optind]);
    const char *pasta_saida = argv[optind + 1];
    char **caminhos_frames = argv + optind + 2;
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
    // O halo não pode ser maior que a menor faixa
    int menor_faixa = altura / num_processos;
    if (bloco > menor_faixa) {
        bloco = menor_faixa;
    }

    AreaTrabalho area; // Reaproveitada em todos os frames
    if (!preparar_area(&area, rank, num_processos, largura, altura, bloco)) {
        fprintf(stderr, "Processo %d: erro de alocação\n", rank);
        MPI_Finalize();
        return 1;
    }
    unsigned char *imagem_resultado = NULL; // Só no processo 0
    if (rank == 0) {
        imagem_resultado = malloc(total_pixels);
        if (!imagem_resultado) {
            fprintf(stderr, "Erro de alocação\n");
            liberar_area(&area);
            free(frame.pixels);
            MPI_Finalize();
            return 1;
        }
    }

    double tempo_distribuicao = 0.0;
    double tempo_calculo = 0.0;
    double tempo_comunicacao = 0.0;
    double tempo_coleta = 0.0;
    double tempo_escrita = 0.0;
    uint32_t checksum = 2166136261u;
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

        executar_blur(&area, largura, rank, num_processos, num_iteracoes, bloco, &tempo_calculo, &tempo_comunicacao);

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
            checksum = atualizar_checksum(checksum, imagem_resultado, total_pixels);
        }
        frames_processados++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tempo_total = agora() - instante_inicio;

    if (rank == 0) {
        printf("Frames: %d de %dx%d pixels\n", frames_processados, largura, altura);
        printf("Vídeo processado em %.6f s (%d frames, %d processos, bloco %d, %d iterações, %.2f frames/s)\n",
               tempo_total, frames_processados, num_processos, bloco, num_iteracoes, frames_processados / tempo_total);
        printf("Tempos do processo 0: leitura %.3f s, distribuição %.3f s, cálculo %.3f s, halos %.3f s, coleta %.3f s, escrita %.3f s\n",
               tempo_leitura, tempo_distribuicao, tempo_calculo, tempo_comunicacao, tempo_coleta, tempo_escrita);
        printf("Checksum do resultado: %08x\n", (unsigned)checksum);
        printf("Imagens salvas em %s/blur_*.pgm\n", pasta_saida);

        if (pasta_referencia != NULL && execucao_com_sucesso) {
            if (!verificar_resultados(pasta_saida, pasta_referencia, frames_processados)) {
                execucao_com_sucesso = 0;
            }
        }
    }

    liberar_area(&area);
    free(imagem_resultado);
    free(frame.pixels);
    MPI_Finalize();
    return execucao_com_sucesso ? 0 : 1;
}
