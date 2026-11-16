// Filtro de Blur 3x3 Iterado - versão PARALELA com MPI (OpenMPI)
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: mpicc -O2 -Wall -Wextra -o blur_par paralelo.c      (ou: make)
// Execução:   mpirun -np P ./blur_par [-b bloco] [-v pasta_referencia]
//                 iteracoes pasta_saida frame0.pgm frame1.pgm ...

// Funções MPI usadas (todas vistas em aula): MPI_Init, MPI_Comm_size,
// MPI_Comm_rank, MPI_Send, MPI_Recv, MPI_Bcast, MPI_Barrier, MPI_Finalize.

// ESTRATÉGIA DE PARALELIZAÇÃO (decomposição de domínio por faixas de linhas)
//   - A imagem é dividida em P faixas de linhas consecutivas, uma por processo
//     (as faixas diferem em no máximo 1 linha de altura).
//   - O processo 0 lê cada frame do disco, envia a faixa de cada processo
//     (MPI_Send/MPI_Recv), calcula a sua própria faixa, recebe as faixas
//     prontas e grava a imagem final.
//   - O filtro 3x3 de um pixel depende só dos vizinhos imediatos. Por isso,
//     cada processo guarda, além da sua faixa, linhas extras ("halo") que são
//     cópias das linhas dos vizinhos de cima e de baixo.
//   - BLOQUEIO TEMPORAL: o halo tem até B linhas (opção -b). A cada "rodada" os
//     processos trocam os halos UMA vez e depois fazem B iterações seguidas
//     sem se comunicar. Cada iteração calcula uma linha a menos em cada ponta
//     do halo (a ponta fica desatualizada), de modo que, após B iterações,
//     sobram exatamente as linhas da faixa, corretas. As linhas do halo
//     calculadas nos dois processos são trabalho redundante, trocado por menos
//     mensagens. Com B = 1 a troca ocorre a cada iteração.
//   - O resultado é IDÊNTICO ao do sequencial (as mesmas contas, na mesma
//     ordem para cada pixel), o que é conferido pela opção -v.
//
// REGRAS DO FILTRO (iguais às do sequencial):
//   - pixel interno = (soma dos 9 vizinhos + 4) / 9, aritmética inteira;
//   - pixels da borda da imagem (linhas 0 e H-1, colunas 0 e W-1) não mudam.

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

// Compatibilidade: no Windows (MinGW) o mkdir recebe só 1 argumento
#ifdef _WIN32
#include <direct.h>
#define CRIAR_PASTA(caminho) _mkdir(caminho)
#else
#define CRIAR_PASTA(caminho) mkdir(caminho, 0755)
#endif

#define BLOCO_PADRAO 4 // Iterações por rodada (e linhas de halo) quando -b não é informado

// Etiquetas (tags) das mensagens MPI: separam os tipos de mensagem
#define TAG_FAIXA 1 // Faixa inicial enviada do processo 0 para os demais
#define TAG_RESULTADO 2 // Faixa final enviada dos demais para o processo 0
#define TAG_HALO_PARA_CIMA 3 // Linhas que viajam para o processo de rank menor
#define TAG_HALO_PARA_BAIXO 4 // Linhas que viajam para o processo de rank maior

// Imagem em tons de cinza (1 byte por pixel), armazenada linha a linha:
// o pixel da linha l, coluna c fica em pixels[l * largura + c]
typedef struct {
    unsigned char *pixels; // Intensidades dos pixels (0 = preto, 255 = branco)
    int largura; // Número de colunas
    int altura; // Número de linhas
} Imagem;

// Área de trabalho de UM processo: a sua faixa mais os halos, em dois buffers
// (uma iteração lê de um e escreve no outro, depois eles trocam de papel).
// Linhas locais: [0, halo_cima) = halo de cima; [halo_cima, halo_cima + num_linhas)
// = linhas da própria faixa; depois vem o halo de baixo.
typedef struct {
    int linha_inicial; // Primeira linha GLOBAL da faixa do processo
    int num_linhas; // Quantidade de linhas da faixa
    int halo_cima; // Linhas de halo acima da faixa (0 se não há vizinho acima)
    int halo_baixo; // Linhas de halo abaixo da faixa (0 se não há vizinho abaixo)
    int linhas_enviar_cima; // Linhas que o processo de cima precisa receber de mim
    int linhas_enviar_baixo; // Linhas que o processo de baixo precisa receber de mim
    int fixa_em_cima; // 1 se a primeira linha local é a linha 0 da imagem (nunca muda)
    int fixa_em_baixo; // 1 se a última linha local é a última da imagem (nunca muda)
    int linhas_locais; // Total de linhas dos buffers (halos + faixa)
    unsigned char *buffer_atual; // Imagem da iteração anterior (lida)
    unsigned char *buffer_novo; // Imagem sendo calculada (escrita)
} AreaTrabalho;

// Objetivo: instante atual em segundos (relógio monotônico), para cronometrar.
// Igual ao agora() do sequencial; não usa nenhuma função MPI fora da lista de aula.
double agora(void) {
    struct timespec instante; // Segundos + nanossegundos
    clock_gettime(CLOCK_MONOTONIC, &instante);
    return instante.tv_sec + instante.tv_nsec / 1e9;
}

// Objetivo: carregar do disco um frame em tons de cinza no formato PGM binário (P5).
// Funcionamento: confere a assinatura "P5", pula espaços e comentários ('#')
// do cabeçalho, lê largura, altura e valor máximo (255) e lê todos os pixels.
// Em caso de erro escreve a mensagem em stderr e devolve 0 (não encerra o
// programa, para que todos os processos possam terminar juntos).
// Retorno: 1 em sucesso (imagem->pixels deve ser liberado por quem chamou).
int ler_pgm(const char *caminho, Imagem *imagem) {
    FILE *arquivo = fopen(caminho, "rb"); // Abre o arquivo em modo binário (leitura)
    if (!arquivo) {
        fprintf(stderr, "Erro ao abrir %s\n", caminho);
        return 0;
    }

    char assinatura[3]; // Guarda os 2 caracteres lidos (esperado: P5)
    if (fscanf(arquivo, "%2s", assinatura) != 1 || strcmp(assinatura, "P5") != 0) {
        fprintf(stderr, "Formato inválido em %s (esperado P5)\n", caminho);
        fclose(arquivo);
        return 0;
    }

    int largura; // Largura lida do cabeçalho
    int altura; // Altura lida do cabeçalho
    int valor_maximo; // Maior intensidade possível, lida do cabeçalho
    int caractere; // Caractere lido do arquivo (int para comportar EOF)
    while (1) { // Pula espaços em branco e linhas de comentário
        caractere = fgetc(arquivo);
        if (caractere == EOF) {
            fprintf(stderr, "Cabeçalho incompleto em %s\n", caminho);
            fclose(arquivo);
            return 0;
        }
        if (caractere == '#') {
            while ((caractere = fgetc(arquivo)) != '\n' && caractere != EOF) {
                // Descarta o resto da linha de comentário
            }
        } else if (caractere != ' ' && caractere != '\t' && caractere != '\n' && caractere != '\r') {
            break;
        }
    }
    ungetc(caractere, arquivo); // Devolve o 1º dígito da largura para o fscanf

    if (fscanf(arquivo, "%d %d %d", &largura, &altura, &valor_maximo) != 3) {
        fprintf(stderr, "Cabeçalho inválido em %s\n", caminho);
        fclose(arquivo);
        return 0;
    }
    fgetc(arquivo); // Consome o \n após o cabeçalho (senão deslocaria a imagem)

    if (largura <= 0 || altura <= 0 || valor_maximo != 255) {
        fprintf(stderr, "PGM não suportado em %s: dimensões positivas e valor máximo 255 são exigidos\n", caminho);
        fclose(arquivo);
        return 0;
    }

    size_t total_pixels = (size_t)largura * altura; // Quantidade de pixels da imagem
    unsigned char *pixels = malloc(total_pixels);
    if (!pixels) {
        fprintf(stderr, "Memória insuficiente para %s\n", caminho);
        fclose(arquivo);
        return 0;
    }
    if (fread(pixels, 1, total_pixels, arquivo) != total_pixels) {
        fprintf(stderr, "Arquivo %s incompleto (faltam pixels)\n", caminho);
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

// Objetivo: gravar a imagem borrada em PGM binário (P5).
// Retorno: 1 em sucesso, 0 em erro (mensagem em stderr).
int escrever_pgm(const char *caminho, const unsigned char *pixels, int largura, int altura) {
    FILE *arquivo = fopen(caminho, "wb"); // Abre arquivo em modo binário (escrita)
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar %s\n", caminho);
        return 0;
    }
    size_t total_pixels = (size_t)largura * altura; // Quantidade de pixels a gravar
    fprintf(arquivo, "P5\n%d %d\n255\n", largura, altura);
    int sucesso = (fwrite(pixels, 1, total_pixels, arquivo) == total_pixels);
    if (fclose(arquivo) != 0) {
        sucesso = 0;
    }
    if (!sucesso) {
        fprintf(stderr, "Erro ao gravar %s\n", caminho);
    }
    return sucesso;
}

// Objetivo: resumo de 32 bits (FNV-1a) do resultado, idêntico ao do sequencial,
// para comparar rapidamente as duas versões. Continua o hash a partir de 'hash'.
uint32_t atualizar_checksum(uint32_t hash, const unsigned char *pixels, size_t total_pixels) {
    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        hash ^= pixels[posicao];
        hash *= 16777619u;
    }
    return hash;
}

// Objetivo: dizer quais linhas da imagem pertencem à faixa de um processo.
// Funcionamento: divide 'altura' linhas entre 'num_processos' processos; os
// primeiros (altura % num_processos) processos recebem uma linha a mais, então
// as faixas diferem em no máximo 1 linha. Qualquer processo consegue calcular
// a faixa de qualquer outro, sem trocar mensagens.
// Parâmetros: rank do processo; saídas: primeira linha global e quantidade de linhas.
void calcular_faixa(int rank, int num_processos, int altura, int *linha_inicial, int *num_linhas) {
    int linhas_base = altura / num_processos; // Linhas que todo processo tem
    int linhas_extras = altura % num_processos; // Processos de rank menor ficam com 1 a mais
    if (rank < linhas_extras) {
        *num_linhas = linhas_base + 1;
        *linha_inicial = rank * (linhas_base + 1);
    } else {
        *num_linhas = linhas_base;
        *linha_inicial = linhas_extras * (linhas_base + 1) + (rank - linhas_extras) * linhas_base;
    }
}

// Objetivo: preencher a área de trabalho de um processo (faixa, halos, buffers).
// Funcionamento: o halo de cima tem min(bloco, linhas acima da faixa) linhas e
// o de baixo min(bloco, linhas abaixo). Se o halo chega até a linha 0 (ou até
// a última linha) da imagem, essa linha é "fixa": o filtro nunca a altera.
// As linhas a enviar a cada vizinho são o tamanho do halo DELE que me cobre,
// calculado pela mesma fórmula (por isso as mensagens sempre casam).
// Pré-condição: bloco <= menor faixa (garantido pela main), para que cada
// vizinho consiga fornecer todas as linhas do halo.
// Retorno: 1 em sucesso, 0 se faltar memória.
int preparar_area(AreaTrabalho *area, int rank, int num_processos, int largura, int altura, int bloco) {
    calcular_faixa(rank, num_processos, altura, &area->linha_inicial, &area->num_linhas);
    int linha_depois_da_faixa = area->linha_inicial + area->num_linhas; // Primeira linha global abaixo da faixa
    int linhas_acima = area->linha_inicial; // Linhas da imagem acima da faixa
    int linhas_abaixo = altura - linha_depois_da_faixa; // Linhas da imagem abaixo da faixa

    area->halo_cima = (linhas_acima < bloco) ? linhas_acima : bloco;
    area->halo_baixo = (linhas_abaixo < bloco) ? linhas_abaixo : bloco;
    // O vizinho de cima tem halo_baixo = min(bloco, linhas abaixo da faixa DELE),
    // e as linhas abaixo da faixa dele são as da minha faixa para baixo
    area->linhas_enviar_cima = (rank > 0) ? ((altura - area->linha_inicial < bloco) ? altura - area->linha_inicial : bloco) : 0;
    // O vizinho de baixo tem halo_cima = min(bloco, linhas acima da faixa DELE)
    area->linhas_enviar_baixo = (rank < num_processos - 1) ? ((linha_depois_da_faixa < bloco) ? linha_depois_da_faixa : bloco) : 0;

    area->fixa_em_cima = (area->linha_inicial - area->halo_cima == 0);
    area->fixa_em_baixo = (linha_depois_da_faixa + area->halo_baixo == altura);
    area->linhas_locais = area->halo_cima + area->num_linhas + area->halo_baixo;

    size_t tamanho_buffer = (size_t)area->linhas_locais * largura; // Bytes de cada buffer
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

// Libera os buffers da área de trabalho.
void liberar_area(AreaTrabalho *area) {
    free(area->buffer_atual);
    free(area->buffer_novo);
    area->buffer_atual = NULL;
    area->buffer_novo = NULL;
}

// Objetivo: uma iteração do blur 3x3 sobre um intervalo de linhas LOCAIS.
// Funcionamento: para cada linha do intervalo, copia os pixels das colunas 0 e
// largura-1 (a borda lateral nunca muda, mas o buffer de destino precisa
// tê-los) e, para as colunas internas, soma as três colunas vizinhas
// reaproveitando as somas já feitas (3 leituras por pixel em vez de 9) e grava
// (soma + 4) / 9. É o mesmo cálculo do sequencial otimizado.
// Parâmetros: origem (lida), destino (escrito), largura, primeira_linha e
// ultima_linha (inclusive, em linhas locais; exigem as linhas vizinhas
// primeira_linha-1 e ultima_linha+1 dentro do buffer).
void filtro_linhas(const unsigned char *origem, unsigned char *destino, int largura, int primeira_linha, int ultima_linha) {
    for (int linha = primeira_linha; linha <= ultima_linha; linha++) { // Linha local atual
        const unsigned char *linha_acima = origem + (size_t)(linha - 1) * largura;
        const unsigned char *linha_atual = origem + (size_t)linha * largura;
        const unsigned char *linha_abaixo = origem + (size_t)(linha + 1) * largura;
        unsigned char *linha_destino = destino + (size_t)linha * largura;

        linha_destino[0] = linha_atual[0]; // Borda esquerda: não muda
        linha_destino[largura - 1] = linha_atual[largura - 1]; // Borda direita: não muda
        if (largura < 3) {
            continue; // Sem colunas internas
        }

        // Somas verticais (3 pixels) das duas primeiras colunas
        int soma_coluna_anterior = linha_acima[0] + linha_atual[0] + linha_abaixo[0];
        int soma_coluna_central = linha_acima[1] + linha_atual[1] + linha_abaixo[1];

        for (int coluna = 1; coluna < largura - 1; coluna++) { // Coluna do pixel atual
            int soma_coluna_seguinte = linha_acima[coluna + 1] + linha_atual[coluna + 1] + linha_abaixo[coluna + 1];
            int soma_vizinhanca = soma_coluna_anterior + soma_coluna_central + soma_coluna_seguinte;
            linha_destino[coluna] = (unsigned char)((soma_vizinhanca + 4) / 9);

            soma_coluna_anterior = soma_coluna_central;
            soma_coluna_central = soma_coluna_seguinte;
        }
    }
}

// Objetivo: troca de linhas com UM vizinho, sem risco de deadlock.
// Funcionamento: MPI_Send/MPI_Recv bloqueantes poderiam travar se os dois
// lados enviassem ao mesmo tempo. Para evitar isso, o processo de MENOR rank
// do par sempre envia primeiro e depois recebe; o de MAIOR rank sempre recebe
// primeiro e depois envia. Essa ordem é a sincronização entre os vizinhos.
// Parâmetros: vizinho (rank), dados a enviar e quantidade de bytes, local onde
// receber e quantidade de bytes, tags de envio e de recebimento e se este
// processo é o de menor rank do par.
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

// Objetivo: atualizar os halos (de cima e de baixo) de um processo com as
// linhas atuais dos vizinhos, e entregar aos vizinhos as minhas linhas.
// Funcionamento: envia ao vizinho de cima as primeiras linhas da minha faixa
// (que formam o halo de baixo dele) e recebe as últimas linhas da faixa dele
// no meu halo de cima; faz o análogo com o vizinho de baixo. Processos de
// rank par conversam primeiro com o de baixo e depois com o de cima; os de
// rank ímpar fazem o contrário. Assim, cada par de vizinhos se encontra no
// mesmo momento e a cadeia de processos nunca fica esperando em círculo.
// Parâmetros: area, buffer a atualizar (o atual), largura, rank, num_processos.
void trocar_halos(const AreaTrabalho *area, unsigned char *buffer, int largura, int rank, int num_processos) {
    // Posições (em linhas locais) das regiões envolvidas
    unsigned char *halo_de_cima = buffer; // Onde chega o halo de cima
    unsigned char *primeiras_proprias = buffer + (size_t)area->halo_cima * largura; // Início da faixa
    unsigned char *ultimas_proprias = buffer + (size_t)(area->halo_cima + area->num_linhas - area->linhas_enviar_baixo) * largura; // Últimas linhas da faixa
    unsigned char *halo_de_baixo = buffer + (size_t)(area->halo_cima + area->num_linhas) * largura; // Onde chega o halo de baixo

    int tem_vizinho_cima = (rank > 0); // 1 se existe processo com rank-1
    int tem_vizinho_baixo = (rank < num_processos - 1); // 1 se existe processo com rank+1

    int conversar_com_baixo_primeiro = (rank % 2 == 0); // Ranks pares: baixo -> cima; ímpares: cima -> baixo
    for (int etapa = 0; etapa < 2; etapa++) {
        int falar_com_baixo = (etapa == 0) ? conversar_com_baixo_primeiro : !conversar_com_baixo_primeiro; // 1 = vizinho de baixo, 0 = de cima
        if (falar_com_baixo && tem_vizinho_baixo) {
            // Eu sou o menor rank do par (rank, rank+1)
            trocar_com_vizinho(rank + 1,
                               ultimas_proprias, area->linhas_enviar_baixo * largura,
                               halo_de_baixo, area->halo_baixo * largura,
                               TAG_HALO_PARA_BAIXO, TAG_HALO_PARA_CIMA, 1);
        } else if (!falar_com_baixo && tem_vizinho_cima) {
            // Eu sou o maior rank do par (rank-1, rank)
            trocar_com_vizinho(rank - 1,
                               primeiras_proprias, area->linhas_enviar_cima * largura,
                               halo_de_cima, area->halo_cima * largura,
                               TAG_HALO_PARA_CIMA, TAG_HALO_PARA_BAIXO, 0);
        }
    }
}

// Objetivo: rodar as N iterações do blur sobre a faixa do processo.
// Funcionamento: repete "rodadas". Em cada rodada: (1) troca os halos; (2)
// copia os halos recebidos para o outro buffer (as linhas fixas e as bordas
// precisam existir nos dois); (3) faz até 'bloco' iterações seguidas, sem
// mensagens. Na iteração m da rodada, o halo desatualizado já cobre m linhas
// em cada ponta, então só se calcula entre a linha m e a linha
// linhas_locais-1-m (linhas fixas da imagem não têm ponta desatualizada e
// começam em 1 / terminam em linhas_locais-2). No fim da rodada sobram
// corretas, no mínimo, as linhas da própria faixa.
// Parâmetros: area (os buffers atual/novo são trocados), largura, rank,
// num_processos, num_iteracoes, bloco; saídas: tempo gasto calculando e tempo
// gasto trocando halos (inclui esperas pelos vizinhos).
void executar_blur(AreaTrabalho *area, int largura, int rank, int num_processos, int num_iteracoes, int bloco,
                   double *tempo_calculo, double *tempo_comunicacao) {
    unsigned char *buffer_atual = area->buffer_atual; // Cópias locais dos ponteiros
    unsigned char *buffer_novo = area->buffer_novo;
    int iteracoes_restantes = num_iteracoes; // Iterações ainda por fazer

    while (iteracoes_restantes > 0) {
        int iteracoes_da_rodada = (iteracoes_restantes < bloco) ? iteracoes_restantes : bloco; // Iterações desta rodada

        double inicio = agora(); // Início da etapa sendo cronometrada
        trocar_halos(area, buffer_atual, largura, rank, num_processos);
        // O outro buffer também precisa das linhas de halo (fixas e iniciais)
        memcpy(buffer_novo, buffer_atual, (size_t)area->halo_cima * largura);
        size_t inicio_halo_baixo = (size_t)(area->halo_cima + area->num_linhas) * largura; // Deslocamento do halo de baixo
        memcpy(buffer_novo + inicio_halo_baixo, buffer_atual + inicio_halo_baixo, (size_t)area->halo_baixo * largura);
        *tempo_comunicacao += agora() - inicio;

        inicio = agora();
        for (int passo = 1; passo <= iteracoes_da_rodada; passo++) { // Iteração m da rodada
            int primeira_linha = area->fixa_em_cima ? 1 : passo; // Primeira linha local calculada
            int ultima_linha = area->fixa_em_baixo ? area->linhas_locais - 2 : area->linhas_locais - 1 - passo; // Última linha local calculada
            filtro_linhas(buffer_atual, buffer_novo, largura, primeira_linha, ultima_linha);

            // Troca de buffers (swap de ponteiros)
            unsigned char *ponteiro_temporario = buffer_atual;
            buffer_atual = buffer_novo;
            buffer_novo = ponteiro_temporario;
        }
        *tempo_calculo += agora() - inicio;

        iteracoes_restantes -= iteracoes_da_rodada;
    }

    area->buffer_atual = buffer_atual; // O resultado final fica em buffer_atual
    area->buffer_novo = buffer_novo;
}

// Objetivo: o processo 0 entrega a cada processo a sua faixa do frame, e
// todos preparam os dois buffers.
// Funcionamento: o processo 0 envia com MPI_Send as linhas de cada faixa; os
// demais recebem com MPI_Recv direto na posição da faixa dentro do buffer
// (depois do halo de cima). Em seguida o buffer novo vira cópia do atual.
// Parâmetros: frame (só o processo 0 usa os pixels), area, largura, altura, rank, num_processos.
void distribuir_faixas(const Imagem *frame, AreaTrabalho *area, int largura, int altura, int rank, int num_processos) {
    size_t deslocamento_faixa = (size_t)area->halo_cima * largura; // Onde a faixa começa no buffer local
    if (rank == 0) {
        for (int destino = 1; destino < num_processos; destino++) {
            int linha_inicial_destino, linhas_destino; // Faixa do processo 'destino'
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

// Objetivo: o processo 0 junta as faixas prontas na imagem final.
// Funcionamento: cada processo envia as linhas da sua faixa (sem os halos)
// com MPI_Send; o processo 0 recebe cada uma direto na posição certa de
// 'resultado' e copia a sua própria.
// Parâmetros: area, resultado (imagem completa; só o processo 0 usa), largura, altura, rank, num_processos.
void coletar_faixas(const AreaTrabalho *area, unsigned char *resultado, int largura, int altura, int rank, int num_processos) {
    size_t deslocamento_faixa = (size_t)area->halo_cima * largura; // Onde a faixa começa no buffer local
    if (rank == 0) {
        for (int origem = 1; origem < num_processos; origem++) {
            int linha_inicial_origem, linhas_origem; // Faixa do processo 'origem'
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

// Objetivo: TESTE DE CORRETUDE, executado só quando o usuário passa -v.
// Funcionamento: depois da medição de tempo, o processo 0 lê, para cada frame,
// o arquivo gerado pelo paralelo (pasta_saida) e o gerado pelo sequencial
// (pasta_referencia) e compara pixel a pixel, mostrando a primeira diferença.
// Retorno: 1 se todos os frames são idênticos, 0 caso contrário.
int verificar_resultados(const char *pasta_saida, const char *pasta_referencia, int num_frames) {
    int tudo_igual = 1; // Vira 0 se algum frame diferir
    for (int indice = 0; indice < num_frames; indice++) {
        char caminho_paralelo[4096]; // Arquivo gerado por este programa
        char caminho_referencia[4096]; // Arquivo gerado pelo sequencial
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
            printf("VERIFICAÇÃO frame %d: FALHA - dimensões diferentes\n", indice);
            tudo_igual = 0;
        } else {
            size_t total_pixels = (size_t)imagem_paralela.largura * imagem_paralela.altura; // Pixels a comparar
            size_t total_diferencas = 0; // Quantos pixels diferem
            size_t primeira_diferenca = 0; // Posição linear do primeiro pixel diferente
            for (size_t posicao = 0; posicao < total_pixels; posicao++) {
                if (imagem_paralela.pixels[posicao] != imagem_referencia.pixels[posicao]) {
                    if (total_diferencas == 0) {
                        primeira_diferenca = posicao;
                    }
                    total_diferencas++;
                }
            }
            if (total_diferencas == 0) {
                printf("VERIFICAÇÃO frame %d: OK - imagens idênticas\n", indice);
            } else {
                printf("VERIFICAÇÃO frame %d: FALHA - %zu pixel(s) diferente(s); primeira diferença em (linha %zu, coluna %zu)\n",
                       indice, total_diferencas, primeira_diferenca / imagem_paralela.largura,
                       primeira_diferenca % imagem_paralela.largura);
                tudo_igual = 0;
            }
        }
        free(imagem_paralela.pixels);
        free(imagem_referencia.pixels);
    }
    printf(tudo_igual ? "VERIFICAÇÃO: TODOS OS FRAMES IDÊNTICOS AO SEQUENCIAL\n" : "VERIFICAÇÃO: HÁ DIFERENÇAS EM RELAÇÃO AO SEQUENCIAL\n");
    return tudo_igual;
}

// Objetivo: aplicar o blur 3x3 iterado a uma sequência de frames com vários
// processos MPI (SPMD: todos executam este mesmo programa).
// Funcionamento:
//   1) todos leem as opções; o processo 0 lê o 1º frame e faz MPI_Bcast das
//      dimensões; todos montam a sua área de trabalho;
//   2) para cada frame: o processo 0 lê e avisa por MPI_Bcast se pode
//      continuar; distribui faixas; todos rodam o blur (com troca de halos);
//      o processo 0 coleta as faixas e grava blur_<i>.pgm;
//   3) MPI_Barrier no início e no fim do cronômetro, para medir o tempo total
//      (leitura + cálculo + escrita), igual ao do sequencial;
//   4) se pedido (-v), o processo 0 compara os resultados com os do sequencial.
// Uso: mpirun -np P ./blur_par [-b bloco] [-v pasta_referencia] iteracoes pasta_saida frame0.pgm ...
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; // Identificador deste processo (0 a num_processos-1)
    int num_processos; // Total de processos do mpirun -np
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

    // ---- Opções e argumentos (todos os processos recebem os mesmos argv) ----
    int bloco = BLOCO_PADRAO; // -b: iterações por rodada
    const char *pasta_referencia = NULL; // -v: pasta com a saída do sequencial
    int opcao; // Letra da opção lida pelo getopt
    int opcoes_validas = 1; // Vira 0 se aparecer uma opção inválida
    if (rank != 0) {
        opterr = 0; // Só o processo 0 imprime mensagens de erro do getopt
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
    int num_frames = argc - optind - 2; // Depois de iteracoes e pasta_saida vêm os frames
    if (!opcoes_validas || num_frames < 1 || bloco < 1) {
        if (rank == 0) {
            fprintf(stderr, "Uso: mpirun -np P %s [-b bloco] [-v pasta_referencia] iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int num_iteracoes = atoi(argv[optind]); // N: número de iterações do blur
    const char *pasta_saida = argv[optind + 1]; // Onde gravar blur_0000.pgm, ...
    char **caminhos_frames = argv + optind + 2; // Arquivos PGM dos frames, em ordem
    if (num_iteracoes <= 0) {
        if (rank == 0) {
            fprintf(stderr, "iteracoes precisa ser um inteiro positivo\n");
        }
        MPI_Finalize();
        return 1;
    }

    // ---- Processo 0: pasta de saída e 1º frame (define as dimensões) ----
    Imagem frame = {NULL, 0, 0}; // Frame atual (só o processo 0 guarda pixels)
    int dimensoes[3] = {0, 0, 0}; // {leitura_ok, largura, altura}, enviado por Bcast
    double tempo_leitura = 0.0; // Soma do tempo lendo frames (processo 0)
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
    int largura = dimensoes[1]; // Número de colunas da imagem
    int altura = dimensoes[2]; // Número de linhas da imagem
    size_t total_pixels = (size_t)largura * altura; // Total de pixels

    // Cada processo precisa de pelo menos 1 linha; o halo não pode ser maior
    // que a menor faixa (senão o vizinho não teria linhas suficientes a enviar)
    if (altura < num_processos) {
        if (rank == 0) {
            fprintf(stderr, "Há mais processos (%d) do que linhas na imagem (%d)\n", num_processos, altura);
        }
        free(frame.pixels);
        MPI_Finalize();
        return 1;
    }
    int menor_faixa = altura / num_processos; // Menor número de linhas de uma faixa
    if (bloco > menor_faixa) {
        bloco = menor_faixa;
    }

    AreaTrabalho area; // Faixa + halos deste processo, reaproveitada em todos os frames
    if (!preparar_area(&area, rank, num_processos, largura, altura, bloco)) {
        fprintf(stderr, "Processo %d: memória insuficiente para os buffers\n", rank);
        MPI_Finalize();
        return 1;
    }
    unsigned char *imagem_resultado = NULL; // Imagem completa final (só o processo 0)
    if (rank == 0) {
        imagem_resultado = malloc(total_pixels);
        if (!imagem_resultado) {
            fprintf(stderr, "Memória insuficiente para a imagem de resultado\n");
            liberar_area(&area);
            free(frame.pixels);
            MPI_Finalize();
            return 1;
        }
    }

    // ---- Laço principal, cronometrado ----
    double tempo_distribuicao = 0.0; // Tempo enviando/recebendo as faixas
    double tempo_calculo = 0.0; // Tempo aplicando o filtro (deste processo)
    double tempo_comunicacao = 0.0; // Tempo trocando halos (inclui esperas)
    double tempo_coleta = 0.0; // Tempo reunindo as faixas no processo 0
    double tempo_escrita = 0.0; // Tempo gravando resultados (processo 0)
    uint32_t checksum = 2166136261u; // Resumo de todos os resultados, em ordem
    int execucao_com_sucesso = 1; // Vira 0 se algo falhar
    int frames_processados = 0; // Quantos frames foram concluídos

    MPI_Barrier(MPI_COMM_WORLD); // Todos começam juntos
    double instante_inicio = agora();

    for (int indice = 0; indice < num_frames; indice++) { // Frame atual
        // Leitura (o processo 0 lê e informa a todos se pode continuar)
        int continuar = 1; // 1 = frame lido e compatível; 0 = erro, todos param
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

        double inicio = agora(); // Início da etapa sendo cronometrada
        distribuir_faixas(&frame, &area, largura, altura, rank, num_processos);
        tempo_distribuicao += agora() - inicio;

        executar_blur(&area, largura, rank, num_processos, num_iteracoes, bloco, &tempo_calculo, &tempo_comunicacao);

        inicio = agora();
        coletar_faixas(&area, imagem_resultado, largura, altura, rank, num_processos);
        tempo_coleta += agora() - inicio;

        if (rank == 0) {
            inicio = agora();
            char caminho[4096]; // Caminho do arquivo de saída
            snprintf(caminho, sizeof(caminho), "%s/blur_%04d.pgm", pasta_saida, indice);
            if (!escrever_pgm(caminho, imagem_resultado, largura, altura)) {
                execucao_com_sucesso = 0;
            }
            tempo_escrita += agora() - inicio;
            checksum = atualizar_checksum(checksum, imagem_resultado, total_pixels);
        }
        frames_processados++;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Todos terminam antes de parar o cronômetro
    double tempo_total = agora() - instante_inicio; // Em segundos

    // ---- Relatório (processo 0) e verificação opcional ----
    if (rank == 0) {
        printf("Frames: %d de %dx%d pixels\n", frames_processados, largura, altura);
        printf("Vídeo processado em %.6f s (%d frames, paralelo MPI, %d processos, bloco %d, %d iterações; %.2f frames/s)\n",
               tempo_total, frames_processados, num_processos, bloco, num_iteracoes, frames_processados / tempo_total);
        printf("Tempo por etapa (processo 0): leitura %.3f s, distribuição %.3f s, cálculo %.3f s, troca de halos %.3f s, coleta %.3f s, escrita %.3f s\n",
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
