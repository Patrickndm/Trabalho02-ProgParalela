/*
 * ============================================================================
 *  FILTRO DE BLUR 3x3 ITERADO - VERSÃO SEQUENCIAL
 * ============================================================================
 *  Instituição : <PREENCHER>
 *  Curso       : <PREENCHER>
 *  Disciplina  : <PREENCHER>   Professor: <PREENCHER>
 *  Integrantes : <NOME 1> (matrícula <PREENCHER>)
 *                <NOME 2> (matrícula <PREENCHER>)
 *
 *  OBJETIVO
 *    Aplicar N vezes um filtro de desfoque (média 3x3) sobre uma imagem em
 *    tons de cinza (matriz de H linhas por W colunas, 1 byte por pixel).
 *    Esta versão serve de base para a versão paralela com MPI e de gabarito
 *    para o teste de corretude.
 *
 *  REGRAS DO FILTRO (devem ser IGUAIS na versão paralela)
 *    - Cada pixel interno passa a valer a média dos 9 pixels da vizinhança
 *      3x3, em aritmética inteira, com arredondamento: (soma + 4) / 9.
 *    - Os pixels da borda (linhas 0 e H-1, colunas 0 e W-1) NÃO são
 *      processados: mantêm o valor original.
 *    - Cada iteração lê de uma imagem e escreve em OUTRA (dois buffers); ao
 *      final da iteração os papéis são trocados. Atualizar no lugar daria
 *      resultado errado. Os buffers são gerenciados por uma área de
 *      trabalho, facilitando a organização e a futura integração com MPI.
 *
 *  COMPILAÇÃO
 *    gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -o sequencial sequencial.c
 *
 *  EXEMPLOS DE USO
 *    ./sequencial --altura 2048 --largura 2048 --iteracoes 50
 *    ./sequencial --altura 8 --largura 8 --iteracoes 1 --imprimir
 *    ./sequencial --altura 512 --largura 512 --iteracoes 10 --saida out.pgm
 *    ./sequencial --entrada foto.pgm --iteracoes 5 --saida out.pgm
 *    ./sequencial --altura 512 --largura 512 --iteracoes 10 \
 *                 --comparar saida_paralela.pgm
 *    ./sequencial --autoteste
 *    ./sequencial --ajuda
 *
 *  OBSERVAÇÃO SOBRE TEMPO
 *    O tempo é medido com clock_gettime(CLOCK_MONOTONIC), apropriado para
 *    medições de duração e comparação de desempenho.
 * ============================================================================
 */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Valores padrão dos parâmetros quando não informados na linha de comando. */
#define ALTURA_PADRAO      512UL
#define LARGURA_PADRAO     512UL
#define ITERACOES_PADRAO   10UL
#define SEMENTE_PADRAO     12345UL

/* Limite de tamanho para imprimir as matrizes na tela (opção --imprimir). */
#define LIMITE_IMPRESSAO   32UL

/* Modos de execução do filtro. */
#define MODO_BASICO        0
#define MODO_OTIMIZADO     1

/* ------------------------------------------------------------------------- */
/*  MEDIÇÃO DE TEMPO                                                         */
/* ------------------------------------------------------------------------- */

/*
 * Função: tempo_atual_em_segundos
 * Objetivo: devolver o instante atual em segundos (com fração), para medir
 *           intervalos por diferença entre duas chamadas.
 * Funcionamento: usa CLOCK_MONOTONIC, que não sofre ajustes do relógio
 *           civil durante a medição. Em caso de falha, encerra o programa.
 */
static double tempo_atual_em_segundos(void)
{
    struct timespec instante;
    if (clock_gettime(CLOCK_MONOTONIC, &instante) != 0) {
        perror("Erro ao consultar CLOCK_MONOTONIC");
        exit(EXIT_FAILURE);
    }
    return (double)instante.tv_sec + (double)instante.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------- */
/*  GERAÇÃO DA IMAGEM SINTÉTICA                                              */
/* ------------------------------------------------------------------------- */

/*
 * Função: gerar_imagem_sintetica
 * Objetivo: preencher a imagem com um padrão determinístico (mesmos
 *           parâmetros = mesma imagem em qualquer execução e máquina).
 * Funcionamento: cada pixel recebe a soma de três componentes:
 *   1) um gradiente diagonal (0 a 127),
 *   2) um tabuleiro de blocos 16x16 (0 ou 64),
 *   3) um ruído de 0 a 63 vindo de um gerador congruencial linear de 32 bits
 *      inicializado com a semente, percorrido em ordem de linhas.
 *   O resultado é limitado a 255.
 * Parâmetros: imagem (saída, altura*largura bytes), altura, largura, semente.
 */
static void gerar_imagem_sintetica(unsigned char *imagem, size_t altura,
                                   size_t largura, uint32_t semente)
{
    uint32_t estado_gerador = semente;
    /* Denominadores dos gradientes; evitam divisão por zero em imagens de
       1 linha ou 1 coluna. */
    size_t divisor_linhas  = (altura  > 1) ? (altura  - 1) : 1;
    size_t divisor_colunas = (largura > 1) ? (largura - 1) : 1;

    for (size_t linha = 0; linha < altura; linha++) {
        for (size_t coluna = 0; coluna < largura; coluna++) {
            /* Componente 1: gradiente diagonal de 0 a 127. */
            unsigned int gradiente =
                (unsigned int)(((linha * 127) / divisor_linhas +
                                (coluna * 127) / divisor_colunas) / 2);

            /* Componente 2: tabuleiro de blocos 16x16. */
            unsigned int tabuleiro =
                (((linha / 16) + (coluna / 16)) % 2 == 0) ? 0u : 64u;

            /* Componente 3: ruído pseudoaleatório determinístico (LCG). */
            estado_gerador = estado_gerador * 1664525u + 1013904223u;
            unsigned int ruido = (estado_gerador >> 24) & 63u;

            unsigned int valor = gradiente + tabuleiro + ruido;
            if (valor > 255u) {
                valor = 255u;
            }
            imagem[linha * largura + coluna] = (unsigned char)valor;
        }
    }
}

/* ------------------------------------------------------------------------- */
/*  LEITURA E GRAVAÇÃO DE ARQUIVOS PGM (formato P5, 8 bits)                  */
/* ------------------------------------------------------------------------- */

/*
 * Função: ler_inteiro_pgm
 * Objetivo: ler o próximo número inteiro do cabeçalho de um PGM, ignorando
 *           espaços e comentários (linhas iniciadas por '#').
 * Funcionamento: pula separadores e comentários, acumula dígitos e consome
 *   o separador que encerra o número. No último campo, preserva a distinção
 *   entre CRLF e o primeiro byte dos pixels para não deslocar a imagem.
 * Retorno: 1 em caso de sucesso, 0 se o número estiver ausente ou inválido.
 */
static int ler_inteiro_pgm(FILE *arquivo, long *valor_lido,
                           int *separador_final)
{
    int caractere = fgetc(arquivo);

    while (caractere != EOF) {
        if (caractere == '#') {
            while (caractere != '\n' && caractere != EOF) {
                caractere = fgetc(arquivo);
            }
        } else if (isspace((unsigned char)caractere)) {
            caractere = fgetc(arquivo);
        } else {
            break;
        }
    }

    if (caractere == EOF || !isdigit((unsigned char)caractere)) {
        return 0;
    }

    long acumulado = 0;
    do {
        int digito = caractere - '0';
        if (acumulado > (LONG_MAX - digito) / 10) {
            return 0;
        }
        acumulado = acumulado * 10 + digito;
        caractere = fgetc(arquivo);
    } while (caractere != EOF && isdigit((unsigned char)caractere));

    if (caractere == EOF || !isspace((unsigned char)caractere)) {
        return 0;
    }

    *valor_lido = acumulado;
    if (separador_final != NULL) {
        *separador_final = caractere;
    }
    return 1;
}

/*
 * Função: ler_pgm
 * Objetivo: carregar uma imagem PGM binária (P5, valor máximo 255).
 * Funcionamento: valida o cabeçalho, aloca a matriz e lê os pixels.
 * Parâmetros: nome_arquivo; imagem (saída: ponteiro alocado, a ser liberado
 *             pelo chamador); altura e largura (saídas).
 * Retorno: 1 em caso de sucesso, 0 em caso de erro (mensagem em stderr).
 */
static int ler_pgm(const char *nome_arquivo, unsigned char **imagem,
                   size_t *altura, size_t *largura)
{
    FILE *arquivo = fopen(nome_arquivo, "rb");
    if (arquivo == NULL) {
        fprintf(stderr, "Erro: não foi possível abrir '%s'.\n", nome_arquivo);
        return 0;
    }

    int primeiro_caractere = fgetc(arquivo);
    int segundo_caractere  = fgetc(arquivo);
    if (primeiro_caractere != 'P' || segundo_caractere != '5') {
        fprintf(stderr, "Erro: '%s' não é um PGM binário (P5).\n",
                nome_arquivo);
        fclose(arquivo);
        return 0;
    }

    long largura_lida = 0, altura_lida = 0, valor_maximo = 0;
    int separador_final = 0;
    if (!ler_inteiro_pgm(arquivo, &largura_lida, NULL) ||
        !ler_inteiro_pgm(arquivo, &altura_lida, NULL) ||
        !ler_inteiro_pgm(arquivo, &valor_maximo, &separador_final)) {
        fprintf(stderr, "Erro: cabeçalho inválido em '%s'.\n", nome_arquivo);
        fclose(arquivo);
        return 0;
    }
    if (largura_lida <= 0 || altura_lida <= 0 || valor_maximo != 255) {
        fprintf(stderr, "Erro: '%s' deve ter dimensões positivas e valor "
                        "máximo 255.\n", nome_arquivo);
        fclose(arquivo);
        return 0;
    }

    /* Em PGM, CRLF é uma única quebra de linha lógica. Consumimos o LF
       complementar, mas não outros espaços: o próximo byte já é pixel. */
    if (separador_final == '\r') {
        int proximo = fgetc(arquivo);
        if (proximo != '\n') {
            if (proximo != EOF) {
                ungetc(proximo, arquivo);
            }
        }
    }

    size_t largura_size = (size_t)largura_lida;
    size_t altura_size = (size_t)altura_lida;
    if (altura_size > SIZE_MAX / largura_size) {
        fprintf(stderr, "Erro: dimensões de '%s' excedem o limite de memória.\n",
                nome_arquivo);
        fclose(arquivo);
        return 0;
    }
    size_t total_pixels = largura_size * altura_size;
    unsigned char *pixels = (unsigned char *)malloc(total_pixels);
    if (pixels == NULL) {
        fprintf(stderr, "Erro: memória insuficiente para a imagem.\n");
        fclose(arquivo);
        return 0;
    }

    if (fread(pixels, 1, total_pixels, arquivo) != total_pixels) {
        fprintf(stderr, "Erro: '%s' tem menos pixels que o declarado.\n",
                nome_arquivo);
        free(pixels);
        fclose(arquivo);
        return 0;
    }

    fclose(arquivo);
    *imagem  = pixels;
    *altura  = (size_t)altura_lida;
    *largura = (size_t)largura_lida;
    return 1;
}

/*
 * Função: gravar_pgm
 * Objetivo: salvar a imagem em um arquivo PGM binário (P5).
 * Retorno: 1 em caso de sucesso, 0 em caso de erro.
 */
static int gravar_pgm(const char *nome_arquivo, const unsigned char *imagem,
                      size_t altura, size_t largura)
{
    FILE *arquivo = fopen(nome_arquivo, "wb");
    if (arquivo == NULL) {
        fprintf(stderr, "Erro: não foi possível criar '%s'.\n", nome_arquivo);
        return 0;
    }

    fprintf(arquivo, "P5\n%zu %zu\n255\n", largura, altura);
    size_t total_pixels = altura * largura;
    int sucesso = (fwrite(imagem, 1, total_pixels, arquivo) == total_pixels);
    if (fclose(arquivo) != 0) {
        sucesso = 0;
    }
    if (!sucesso) {
        fprintf(stderr, "Erro ao gravar '%s'.\n", nome_arquivo);
    }
    return sucesso;
}

/* ------------------------------------------------------------------------- */
/*  FILTRO DE BLUR 3x3                                                       */
/* ------------------------------------------------------------------------- */

/*
 * Função: filtro_basico_uma_iteracao
 * Objetivo: calcular UMA iteração do blur 3x3 de forma direta (referência).
 * Funcionamento: para cada pixel interno, soma os 9 vizinhos com dois laços
 *   e grava (soma + 4) / 9 no buffer de destino. A borda não é escrita
 *   (ambos os buffers já a contêm com o valor original).
 * Parâmetros: origem (lida), destino (escrito), altura, largura.
 */
static void filtro_basico_uma_iteracao(const unsigned char *origem,
                                       unsigned char *destino,
                                       size_t altura, size_t largura)
{
    if (altura < 3 || largura < 3) {
        return; /* não há pixel interno: nada a fazer */
    }

    for (size_t linha = 1; linha + 1 < altura; linha++) {
        for (size_t coluna = 1; coluna + 1 < largura; coluna++) {
            int soma_vizinhanca = 0;
            for (size_t linha_vizinha = linha - 1; linha_vizinha <= linha + 1;
                 linha_vizinha++) {
                for (size_t coluna_vizinha = coluna - 1;
                     coluna_vizinha <= coluna + 1; coluna_vizinha++) {
                    soma_vizinhanca +=
                        origem[linha_vizinha * largura + coluna_vizinha];
                }
            }
            destino[linha * largura + coluna] =
                (unsigned char)((soma_vizinhanca + 4) / 9);
        }
    }
}

/*
 * Função: filtro_otimizado_uma_iteracao
 * Objetivo: calcular UMA iteração do blur 3x3 com resultado IDÊNTICO ao da
 *           versão básica, porém mais rápido.
 * Funcionamento: percorre cada linha com três ponteiros (linha acima, atual
 *   e abaixo) e reaproveita as somas de coluna: ao avançar um pixel, só a
 *   soma da nova coluna é calculada; as duas anteriores são reutilizadas.
 *   Reduz de 9 para 3 leituras por pixel e evita recalcular índices.
 * Parâmetros: origem (lida), destino (escrito), altura, largura.
 */
static void filtro_otimizado_uma_iteracao(const unsigned char *origem,
                                          unsigned char *destino,
                                          size_t altura, size_t largura)
{
    if (altura < 3 || largura < 3) {
        return;
    }

    for (size_t linha = 1; linha + 1 < altura; linha++) {
        const unsigned char *linha_acima  = origem + (linha - 1) * largura;
        const unsigned char *linha_atual  = origem + linha * largura;
        const unsigned char *linha_abaixo = origem + (linha + 1) * largura;
        unsigned char *linha_destino      = destino + linha * largura;

        /* Somas verticais (3 pixels) das colunas já conhecidas. */
        int soma_coluna_anterior = linha_acima[0] + linha_atual[0] +
                                   linha_abaixo[0];
        int soma_coluna_central  = linha_acima[1] + linha_atual[1] +
                                   linha_abaixo[1];

        for (size_t coluna = 1; coluna + 1 < largura; coluna++) {
            int soma_coluna_seguinte = linha_acima[coluna + 1] +
                                       linha_atual[coluna + 1] +
                                       linha_abaixo[coluna + 1];
            int soma_vizinhanca = soma_coluna_anterior +
                                  soma_coluna_central +
                                  soma_coluna_seguinte;
            linha_destino[coluna] = (unsigned char)((soma_vizinhanca + 4) / 9);

            soma_coluna_anterior = soma_coluna_central;
            soma_coluna_central  = soma_coluna_seguinte;
        }
    }
}

/*
 * Função: aplicar_filtro_iterado
 * Objetivo: aplicar o blur N vezes alternando dois buffers.
 * Funcionamento: a cada iteração lê de 'buffer_atual' e escreve em
 *   'buffer_proximo'; depois troca os ponteiros. PRÉ-CONDIÇÃO: os dois
 *   buffers começam com o mesmo conteúdo (a borda nunca é reescrita).
 * Parâmetros: os dois buffers, altura, largura, número de iterações e modo
 *   (MODO_BASICO ou MODO_OTIMIZADO).
 * Retorno: ponteiro para o buffer que contém o resultado final (é um dos
 *   dois buffers recebidos).
 */
static unsigned char *aplicar_filtro_iterado(unsigned char *buffer_atual,
                                             unsigned char *buffer_proximo,
                                             size_t altura, size_t largura,
                                             unsigned long iteracoes,
                                             int modo)
{
    for (unsigned long iteracao = 0; iteracao < iteracoes; iteracao++) {
        if (modo == MODO_OTIMIZADO) {
            filtro_otimizado_uma_iteracao(buffer_atual, buffer_proximo,
                                          altura, largura);
        } else {
            filtro_basico_uma_iteracao(buffer_atual, buffer_proximo,
                                       altura, largura);
        }
        unsigned char *ponteiro_auxiliar = buffer_atual;
        buffer_atual   = buffer_proximo;
        buffer_proximo = ponteiro_auxiliar;
    }
    return buffer_atual;
}

/* ------------------------------------------------------------------------- */
/*  VERIFICAÇÃO E UTILITÁRIOS                                                */
/* ------------------------------------------------------------------------- */

/*
 * Função: comparar_imagens
 * Objetivo: comparar duas imagens do mesmo tamanho byte a byte.
 * Funcionamento: percorre todos os pixels contando as diferenças e guarda a
 *   posição (linha, coluna) da primeira diferença encontrada.
 * Parâmetros: imagens a e b, altura, largura; primeira_linha e
 *   primeira_coluna (saídas, válidas só se o retorno for maior que zero).
 * Retorno: número de pixels diferentes (0 = imagens idênticas).
 */
static size_t comparar_imagens(const unsigned char *imagem_a,
                               const unsigned char *imagem_b,
                               size_t altura, size_t largura,
                               size_t *primeira_linha,
                               size_t *primeira_coluna)
{
    size_t total_diferencas = 0;
    size_t total_pixels = altura * largura;

    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        if (imagem_a[posicao] != imagem_b[posicao]) {
            if (total_diferencas == 0) {
                *primeira_linha  = posicao / largura;
                *primeira_coluna = posicao % largura;
            }
            total_diferencas++;
        }
    }
    return total_diferencas;
}

/*
 * Função: calcular_checksum
 * Objetivo: gerar um resumo de 32 bits da imagem (hash FNV-1a). Permite
 *           comparar rapidamente resultados sem trocar arquivos: imagens
 *           iguais têm o mesmo valor.
 */
static uint32_t calcular_checksum(const unsigned char *imagem,
                                  size_t total_pixels)
{
    uint32_t valor_hash = 2166136261u;
    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        valor_hash ^= imagem[posicao];
        valor_hash *= 16777619u;
    }
    return valor_hash;
}

/*
 * Função: imprimir_matriz
 * Objetivo: mostrar a imagem na tela (apenas imagens pequenas), útil para
 *           depurar com imagens 8x8.
 */
static void imprimir_matriz(const char *titulo, const unsigned char *imagem,
                            size_t altura, size_t largura)
{
    if (altura > LIMITE_IMPRESSAO || largura > LIMITE_IMPRESSAO) {
        printf("%s: imagem grande demais para imprimir (limite %lux%lu).\n",
               titulo, LIMITE_IMPRESSAO, LIMITE_IMPRESSAO);
        return;
    }
    printf("%s:\n", titulo);
    for (size_t linha = 0; linha < altura; linha++) {
        for (size_t coluna = 0; coluna < largura; coluna++) {
            printf("%4u", (unsigned int)imagem[linha * largura + coluna]);
        }
        printf("\n");
    }
}

/*
 * Função: copiar_imagem
 * Objetivo: alocar um novo buffer com cópia da imagem recebida.
 * Retorno: ponteiro alocado (liberar com free) ou NULL se faltar memória.
 */
static unsigned char *copiar_imagem(const unsigned char *imagem,
                                    size_t total_pixels)
{
    unsigned char *copia = (unsigned char *)malloc(total_pixels);
    if (copia != NULL) {
        memcpy(copia, imagem, total_pixels);
    }
    return copia;
}

/* ------------------------------------------------------------------------- */
/*  AUTOTESTE                                                                */
/* ------------------------------------------------------------------------- */

/*
 * Função: bordas_iguais
 * Objetivo: verificar se as bordas de duas imagens são idênticas (regra: a
 *           borda nunca é alterada pelo filtro).
 * Retorno: 1 se iguais, 0 caso contrário.
 */
static int bordas_iguais(const unsigned char *imagem_a,
                         const unsigned char *imagem_b,
                         size_t altura, size_t largura)
{
    for (size_t linha = 0; linha < altura; linha++) {
        for (size_t coluna = 0; coluna < largura; coluna++) {
            int pixel_de_borda = (linha == 0 || linha == altura - 1 ||
                                  coluna == 0 || coluna == largura - 1);
            if (pixel_de_borda &&
                imagem_a[linha * largura + coluna] !=
                imagem_b[linha * largura + coluna]) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * Função: executar_um_caso_de_teste
 * Objetivo: testar uma configuração (altura, largura, iterações, semente).
 * Verificações: (1) versão básica e otimizada geram imagens idênticas;
 *   (2) as bordas permanecem iguais às originais; (3) com 0 iterações a
 *   imagem não muda.
 * Retorno: 1 se todas as verificações passaram, 0 caso contrário.
 */
static int executar_um_caso_de_teste(size_t altura, size_t largura,
                                     unsigned long iteracoes,
                                     uint32_t semente)
{
    size_t total_pixels = altura * largura;
    unsigned char *imagem_original = (unsigned char *)malloc(total_pixels);
    if (imagem_original == NULL) {
        return 0;
    }
    gerar_imagem_sintetica(imagem_original, altura, largura, semente);

    unsigned char *basico_a     = copiar_imagem(imagem_original, total_pixels);
    unsigned char *basico_b     = copiar_imagem(imagem_original, total_pixels);
    unsigned char *otimizado_a  = copiar_imagem(imagem_original, total_pixels);
    unsigned char *otimizado_b  = copiar_imagem(imagem_original, total_pixels);
    unsigned char *sem_iteracao_a = copiar_imagem(imagem_original, total_pixels);
    unsigned char *sem_iteracao_b = copiar_imagem(imagem_original, total_pixels);
    int sucesso = 0;

    if (basico_a && basico_b && otimizado_a && otimizado_b &&
        sem_iteracao_a && sem_iteracao_b) {
        unsigned char *resultado_basico = aplicar_filtro_iterado(
            basico_a, basico_b, altura, largura, iteracoes, MODO_BASICO);
        unsigned char *resultado_otimizado = aplicar_filtro_iterado(
            otimizado_a, otimizado_b, altura, largura, iteracoes,
            MODO_OTIMIZADO);
        unsigned char *resultado_zero = aplicar_filtro_iterado(
            sem_iteracao_a, sem_iteracao_b, altura, largura, 0,
            MODO_OTIMIZADO);

        size_t linha_dif = 0, coluna_dif = 0;
        int modos_iguais =
            comparar_imagens(resultado_basico, resultado_otimizado, altura,
                             largura, &linha_dif, &coluna_dif) == 0;
        int bordas_preservadas =
            bordas_iguais(resultado_basico, imagem_original, altura, largura);
        int zero_iteracoes_inalterada =
            comparar_imagens(resultado_zero, imagem_original, altura, largura,
                             &linha_dif, &coluna_dif) == 0;

        sucesso = modos_iguais && bordas_preservadas &&
                  zero_iteracoes_inalterada;
        printf("  %4zux%-4zu iter=%-3lu semente=%-6lu  básico=otimizado:%s  "
               "bordas:%s  N=0:%s\n",
               altura, largura, iteracoes, (unsigned long)semente,
               modos_iguais ? "OK" : "FALHA",
               bordas_preservadas ? "OK" : "FALHA",
               zero_iteracoes_inalterada ? "OK" : "FALHA");
    }

    free(imagem_original);
    free(basico_a);
    free(basico_b);
    free(otimizado_a);
    free(otimizado_b);
    free(sem_iteracao_a);
    free(sem_iteracao_b);
    return sucesso;
}

/*
 * Função: executar_teste_imagem_constante
 * Objetivo: verificar uma propriedade matemática do filtro: uma imagem de
 *           valor constante deve permanecer constante (com o arredondamento
 *           (soma + 4) / 9, 9*c é devolvido exatamente como c).
 * Retorno: 1 se passou, 0 caso contrário.
 */
static int executar_teste_imagem_constante(void)
{
    size_t altura = 9, largura = 11, total_pixels = altura * largura;
    unsigned char *buffer_a = (unsigned char *)malloc(total_pixels);
    unsigned char *buffer_b = (unsigned char *)malloc(total_pixels);
    int sucesso = 0;

    if (buffer_a && buffer_b) {
        memset(buffer_a, 77, total_pixels);
        memset(buffer_b, 77, total_pixels);
        unsigned char *resultado = aplicar_filtro_iterado(
            buffer_a, buffer_b, altura, largura, 7, MODO_OTIMIZADO);
        sucesso = 1;
        for (size_t posicao = 0; posicao < total_pixels; posicao++) {
            if (resultado[posicao] != 77) {
                sucesso = 0;
                break;
            }
        }
    }
    printf("  imagem constante permanece constante: %s\n",
           sucesso ? "OK" : "FALHA");
    free(buffer_a);
    free(buffer_b);
    return sucesso;
}

/*
 * Função: executar_autoteste
 * Objetivo: rodar uma bateria de testes internos, incluindo tamanhos
 *           mínimos e não divisíveis (úteis para depois testar o paralelo).
 * Retorno: 0 se todos passaram, 1 caso contrário (código de saída).
 */
static int executar_autoteste(void)
{
    /* Cada linha: altura, largura, iterações, semente. */
    static const unsigned long casos[][4] = {
        {1, 1, 3, 1},   {1, 9, 2, 2},   {9, 1, 2, 3},   {2, 5, 2, 4},
        {3, 3, 1, 5},   {3, 10, 5, 6},  {8, 8, 0, 7},   {8, 8, 1, 7},
        {8, 8, 4, 7},   {10, 7, 6, 8},  {17, 33, 5, 9}, {64, 64, 10, 10},
        {101, 53, 7, 11}, {256, 300, 3, 12}
    };
    size_t quantidade_casos = sizeof(casos) / sizeof(casos[0]);
    int falhas = 0;

    printf("Autoteste do filtro sequencial:\n");
    for (size_t indice = 0; indice < quantidade_casos; indice++) {
        if (!executar_um_caso_de_teste(casos[indice][0], casos[indice][1],
                                       casos[indice][2],
                                       (uint32_t)casos[indice][3])) {
            falhas++;
        }
    }
    if (!executar_teste_imagem_constante()) {
        falhas++;
    }

    if (falhas == 0) {
        printf("Resultado: TODOS OS TESTES PASSARAM.\n");
        return 0;
    }
    printf("Resultado: %d teste(s) com FALHA.\n", falhas);
    return 1;
}

/* ------------------------------------------------------------------------- */
/*  ARGUMENTOS DA LINHA DE COMANDO                                           */
/* ------------------------------------------------------------------------- */

/* Parâmetros de execução obtidos da linha de comando. */
typedef struct {
    size_t altura;                 /* número de linhas da imagem            */
    size_t largura;                /* número de colunas da imagem           */
    unsigned long iteracoes;       /* quantas vezes aplicar o filtro        */
    uint32_t semente;              /* semente do gerador de imagem          */
    const char *arquivo_entrada;   /* PGM de entrada (NULL = gerar imagem)  */
    const char *arquivo_saida;     /* PGM de saída (NULL = não gravar)      */
    const char *arquivo_comparar;  /* PGM a comparar com o resultado        */
    int modo;                      /* MODO_BASICO ou MODO_OTIMIZADO         */
    int imprimir_matriz;           /* 1 = imprimir imagens pequenas         */
    int saida_csv;                 /* 1 = imprimir uma linha CSV            */
    int executar_autoteste;        /* 1 = rodar o autoteste e sair          */
    int mostrar_ajuda;             /* 1 = mostrar ajuda e sair              */
} Parametros;

/*
 * Função: mostrar_ajuda
 * Objetivo: listar as opções aceitas pelo programa.
 */
static void mostrar_ajuda(const char *nome_programa)
{
    printf("Uso: %s [opções]\n\n", nome_programa);
    printf("  --altura N       número de linhas (padrão %lu)\n", ALTURA_PADRAO);
    printf("  --largura N      número de colunas (padrão %lu)\n",
           LARGURA_PADRAO);
    printf("  --iteracoes N    iterações do filtro (padrão %lu)\n",
           ITERACOES_PADRAO);
    printf("  --semente N      semente do gerador (padrão %lu)\n",
           SEMENTE_PADRAO);
    printf("  --entrada ARQ    ler imagem PGM (P5) em vez de gerar\n");
    printf("  --saida ARQ      gravar o resultado em PGM\n");
    printf("  --comparar ARQ   comparar o resultado com outro PGM "
           "(ex.: saída da versão paralela)\n");
    printf("  --modo M         'otimizado' (padrão) ou 'basico'\n");
    printf("  --imprimir       imprimir matrizes (imagens até %lux%lu)\n",
           LIMITE_IMPRESSAO, LIMITE_IMPRESSAO);
    printf("  --csv            imprimir também uma linha CSV com os tempos\n");
    printf("  --autoteste      executar testes internos e sair\n");
    printf("  --ajuda          mostrar esta ajuda\n");
}

/*
 * Função: converter_para_inteiro
 * Objetivo: converter texto em inteiro sem sinal, rejeitando valores
 *           inválidos (letras, negativos, vazio).
 * Retorno: 1 em caso de sucesso, 0 em caso de erro.
 */
static int converter_para_inteiro(const char *texto, unsigned long *valor)
{
    char *fim_da_conversao = NULL;
    if (texto == NULL || texto[0] == '\0' || texto[0] == '-') {
        return 0;
    }
    errno = 0;
    unsigned long convertido = strtoul(texto, &fim_da_conversao, 10);
    if (errno == ERANGE || fim_da_conversao == texto ||
        *fim_da_conversao != '\0') {
        return 0;
    }
    *valor = convertido;
    return 1;
}

/*
 * Função: processar_argumentos
 * Objetivo: preencher a estrutura de parâmetros a partir de argc/argv.
 * Retorno: 1 em caso de sucesso, 0 se algum argumento for inválido
 *          (mensagem em stderr).
 */
static int processar_argumentos(int argc, char **argv, Parametros *parametros)
{
    parametros->altura           = ALTURA_PADRAO;
    parametros->largura          = LARGURA_PADRAO;
    parametros->iteracoes        = ITERACOES_PADRAO;
    parametros->semente          = (uint32_t)SEMENTE_PADRAO;
    parametros->arquivo_entrada  = NULL;
    parametros->arquivo_saida    = NULL;
    parametros->arquivo_comparar = NULL;
    parametros->modo             = MODO_OTIMIZADO;
    parametros->imprimir_matriz  = 0;
    parametros->saida_csv        = 0;
    parametros->executar_autoteste = 0;
    parametros->mostrar_ajuda    = 0;

    for (int indice = 1; indice < argc; indice++) {
        const char *opcao = argv[indice];
        int precisa_valor =
            strcmp(opcao, "--altura") == 0 || strcmp(opcao, "--largura") == 0 ||
            strcmp(opcao, "--iteracoes") == 0 ||
            strcmp(opcao, "--semente") == 0 ||
            strcmp(opcao, "--entrada") == 0 || strcmp(opcao, "--saida") == 0 ||
            strcmp(opcao, "--comparar") == 0 || strcmp(opcao, "--modo") == 0;

        if (strcmp(opcao, "--imprimir") == 0) {
            parametros->imprimir_matriz = 1;
        } else if (strcmp(opcao, "--csv") == 0) {
            parametros->saida_csv = 1;
        } else if (strcmp(opcao, "--autoteste") == 0) {
            parametros->executar_autoteste = 1;
        } else if (strcmp(opcao, "--ajuda") == 0) {
            parametros->mostrar_ajuda = 1;
        } else if (precisa_valor) {
            if (indice + 1 >= argc) {
                fprintf(stderr, "Erro: a opção %s precisa de um valor.\n",
                        opcao);
                return 0;
            }
            const char *valor_texto = argv[++indice];
            unsigned long valor_numerico = 0;

            if (strcmp(opcao, "--entrada") == 0) {
                parametros->arquivo_entrada = valor_texto;
            } else if (strcmp(opcao, "--saida") == 0) {
                parametros->arquivo_saida = valor_texto;
            } else if (strcmp(opcao, "--comparar") == 0) {
                parametros->arquivo_comparar = valor_texto;
            } else if (strcmp(opcao, "--modo") == 0) {
                if (strcmp(valor_texto, "basico") == 0) {
                    parametros->modo = MODO_BASICO;
                } else if (strcmp(valor_texto, "otimizado") == 0) {
                    parametros->modo = MODO_OTIMIZADO;
                } else {
                    fprintf(stderr, "Erro: modo '%s' inválido (use 'basico' "
                                    "ou 'otimizado').\n", valor_texto);
                    return 0;
                }
            } else {
                if (!converter_para_inteiro(valor_texto, &valor_numerico)) {
                    fprintf(stderr, "Erro: valor '%s' inválido para %s.\n",
                            valor_texto, opcao);
                    return 0;
                }
                if (strcmp(opcao, "--altura") == 0) {
                    if (valor_numerico > SIZE_MAX) {
                        fprintf(stderr, "Erro: altura excede o limite de size_t.\n");
                        return 0;
                    }
                    parametros->altura = (size_t)valor_numerico;
                } else if (strcmp(opcao, "--largura") == 0) {
                    if (valor_numerico > SIZE_MAX) {
                        fprintf(stderr, "Erro: largura excede o limite de size_t.\n");
                        return 0;
                    }
                    parametros->largura = (size_t)valor_numerico;
                } else if (strcmp(opcao, "--iteracoes") == 0) {
                    parametros->iteracoes = valor_numerico;
                } else {
                    if (valor_numerico > UINT32_MAX) {
                        fprintf(stderr, "Erro: a semente deve estar entre 0 e %" PRIu32 ".\n",
                                UINT32_MAX);
                        return 0;
                    }
                    parametros->semente = (uint32_t)valor_numerico;
                }
            }
        } else {
            fprintf(stderr, "Erro: opção desconhecida '%s' (use --ajuda).\n",
                    opcao);
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------- */
/*  ÁREA DE TRABALHO                                                         */
/* ------------------------------------------------------------------------- */

/*
 * A área de trabalho reúne os dois buffers usados pelo filtro iterativo.
 * Essa organização deixa explícita a propriedade fundamental do algoritmo:
 * uma iteração lê de um buffer e escreve no outro, sem atualização in-place.
 */
typedef struct {
    unsigned char *buffer_a;
    unsigned char *buffer_b;
    size_t total_pixels;
} AreaTrabalho;

/*
 * Inicializa os buffers a partir da imagem original. Os dois recebem uma
 * cópia inicial para que as bordas, que nunca são processadas, permaneçam
 * corretas mesmo quando os ponteiros são alternados.
 */
static int inicializar_area_trabalho(AreaTrabalho *area,
                                     const unsigned char *imagem_inicial,
                                     size_t total_pixels)
{
    area->buffer_a = NULL;
    area->buffer_b = NULL;
    area->total_pixels = total_pixels;

    area->buffer_a = copiar_imagem(imagem_inicial, total_pixels);
    area->buffer_b = copiar_imagem(imagem_inicial, total_pixels);

    if (area->buffer_a == NULL || area->buffer_b == NULL) {
        free(area->buffer_a);
        free(area->buffer_b);
        area->buffer_a = NULL;
        area->buffer_b = NULL;
        area->total_pixels = 0;
        return 0;
    }
    return 1;
}

/* Libera todos os recursos pertencentes à área de trabalho. */
static void liberar_area_trabalho(AreaTrabalho *area)
{
    free(area->buffer_a);
    free(area->buffer_b);
    area->buffer_a = NULL;
    area->buffer_b = NULL;
    area->total_pixels = 0;
}

/* ------------------------------------------------------------------------- */
/*  PROGRAMA PRINCIPAL                                                       */
/* ------------------------------------------------------------------------- */

/*
 * Função: main
 * Objetivo: coordenar a execução: ler argumentos, obter a imagem, aplicar o
 *           filtro medindo o tempo, gravar/comparar e relatar os resultados.
 * Código de saída: 0 em sucesso; 1 em erro ou se --comparar encontrar
 *   diferenças.
 */
int main(int argc, char **argv)
{
    Parametros parametros;
    if (!processar_argumentos(argc, argv, &parametros)) {
        return 1;
    }
    if (parametros.mostrar_ajuda) {
        mostrar_ajuda(argv[0]);
        return 0;
    }
    if (parametros.executar_autoteste) {
        return executar_autoteste();
    }

    /* ---- Etapa 1: obter a imagem inicial (gerar ou ler) ---------------- */
    unsigned char *imagem_inicial = NULL;
    size_t altura = parametros.altura;
    size_t largura = parametros.largura;
    double instante_inicio_entrada = tempo_atual_em_segundos();

    if (parametros.arquivo_entrada != NULL) {
        if (!ler_pgm(parametros.arquivo_entrada, &imagem_inicial, &altura,
                     &largura)) {
            return 1;
        }
    } else {
        if (altura == 0 || largura == 0) {
            fprintf(stderr, "Erro: altura e largura devem ser maiores que "
                            "zero.\\n");
            return 1;
        }
        if (altura > SIZE_MAX / largura) {
            fprintf(stderr, "Erro: tamanho da imagem excede o limite.\\n");
            return 1;
        }

        size_t total = altura * largura;
        imagem_inicial = (unsigned char *)malloc(total);
        if (imagem_inicial == NULL) {
            fprintf(stderr, "Erro: memória insuficiente.\\n");
            return 1;
        }
        gerar_imagem_sintetica(imagem_inicial, altura, largura,
                               parametros.semente);
    }

    double tempo_entrada = tempo_atual_em_segundos() -
                           instante_inicio_entrada;
    size_t total_pixels = altura * largura;

    /* ---- Etapa 2: preparar a área de trabalho -------------------------- */
    AreaTrabalho area;
    if (!inicializar_area_trabalho(&area, imagem_inicial, total_pixels)) {
        fprintf(stderr, "Erro: não foi possível alocar os buffers de trabalho.\\n");
        free(imagem_inicial);
        return 1;
    }
    free(imagem_inicial);
    imagem_inicial = NULL;

    if (parametros.imprimir_matriz) {
        imprimir_matriz("Imagem inicial", area.buffer_a, altura, largura);
    }

    /* ---- Etapa 3: aplicar o filtro (tempo usado no speedup) ------------- */
    double instante_inicio_filtro = tempo_atual_em_segundos();
    unsigned char *resultado = aplicar_filtro_iterado(
        area.buffer_a, area.buffer_b, altura, largura, parametros.iteracoes,
        parametros.modo);
    double tempo_filtro = tempo_atual_em_segundos() -
                          instante_inicio_filtro;

    if (parametros.imprimir_matriz) {
        imprimir_matriz("Imagem final", resultado, altura, largura);
    }

    /* ---- Etapa 4: gravar a imagem final (opcional) --------------------- */
    double tempo_gravacao = 0.0;
    int execucao_com_sucesso = 1;
    if (parametros.arquivo_saida != NULL) {
        double instante_inicio_gravacao = tempo_atual_em_segundos();
        if (!gravar_pgm(parametros.arquivo_saida, resultado, altura,
                        largura)) {
            execucao_com_sucesso = 0;
        }
        tempo_gravacao = tempo_atual_em_segundos() -
                         instante_inicio_gravacao;
    }

    /* ---- Etapa 5: relatório dos resultados ----------------------------- */
    uint32_t checksum_resultado = calcular_checksum(resultado, total_pixels);
    printf("Modo: %s | Imagem: %zux%zu | Iterações: %lu | Semente: %lu\\n",
           parametros.modo == MODO_OTIMIZADO ? "otimizado" : "básico",
           altura, largura, parametros.iteracoes,
           (unsigned long)parametros.semente);
    printf("Tempo de geração/leitura : %.6f s\\n", tempo_entrada);
    printf("Tempo do filtro          : %.6f s\\n", tempo_filtro);
    if (parametros.arquivo_saida != NULL) {
        printf("Tempo de gravação        : %.6f s\\n", tempo_gravacao);
    }
    printf("Checksum do resultado    : %08lx\\n",
           (unsigned long)checksum_resultado);

    if (parametros.saida_csv) {
        printf("CSV,%s,%zu,%zu,%lu,%lu,%.6f,%.6f,%.6f,%08lx\\n",
               parametros.modo == MODO_OTIMIZADO ? "otimizado" : "basico",
               altura, largura, parametros.iteracoes,
               (unsigned long)parametros.semente, tempo_entrada,
               tempo_filtro, tempo_gravacao,
               (unsigned long)checksum_resultado);
    }

    /* ---- Etapa 6: comparação com outro PGM (opcional) ------------------ */
    if (parametros.arquivo_comparar != NULL) {
        unsigned char *imagem_para_comparar = NULL;
        size_t altura_comparada = 0, largura_comparada = 0;
        if (!ler_pgm(parametros.arquivo_comparar, &imagem_para_comparar,
                     &altura_comparada, &largura_comparada)) {
            execucao_com_sucesso = 0;
        } else if (altura_comparada != altura ||
                   largura_comparada != largura) {
            printf("COMPARAÇÃO: FALHA - dimensões diferentes (%zux%zu contra "
                   "%zux%zu).\\n", altura, largura, altura_comparada,
                   largura_comparada);
            execucao_com_sucesso = 0;
        } else {
            size_t linha_dif = 0, coluna_dif = 0;
            size_t total_diferencas = comparar_imagens(
                resultado, imagem_para_comparar, altura, largura,
                &linha_dif, &coluna_dif);
            if (total_diferencas == 0) {
                printf("COMPARAÇÃO: OK - imagens idênticas.\\n");
            } else {
                printf("COMPARAÇÃO: FALHA - %zu pixel(s) diferente(s); "
                       "primeira diferença em (linha %zu, coluna %zu).\\n",
                       total_diferencas, linha_dif, coluna_dif);
                execucao_com_sucesso = 0;
            }
        }
        free(imagem_para_comparar);
    }

    liberar_area_trabalho(&area);
    return execucao_com_sucesso ? 0 : 1;
}
