// Filtro de Blur 3x3 Iterado - versão PARALELA com MPI (OpenMPI)
// Universidade Estadual de Maringá (UEM)
// Curso: Ciência da Computação (CC)
// Disciplina: Programação Paralela - DIN12041
// Professor: Ronaldo Augusto de Lara Goncalves

// Integrantes:
// Matheus Foltran Consonni - RA 133862
// Patrick Nelson de Morais - RA 130166

// Compilação: make   (ou: gcc -O2 -o blur_seq sequencial.c)
// Uso: [-b] iteracoes pasta_saida frame0.pgm frame1.pgm ... frameN-1.pgm
//   -b  usa o filtro básico (9 leituras por pixel) em vez do otimizado

// Regras do filtro (devem ser IGUAIS na versão paralela):
//   - pixel interno = (soma dos 9 vizinhos + 4) / 9, aritmética inteira;
//   - pixels da borda (linhas 0 e H-1, colunas 0 e W-1) mantêm o valor original;
//   - cada iteração lê de um buffer e escreve em OUTRO (sem atualizar no lugar).

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
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

#define MODO_BASICO 0
#define MODO_OTIMIZADO 1

// Imagem em tons de cinza (1 byte por pixel), armazenada linha a linha:
// o pixel da linha l, coluna c fica em pixels[l * largura + c]
typedef struct {
    unsigned char *pixels; // Intensidades dos pixels (0 = preto, 255 = branco)
    int largura; // Número de colunas
    int altura; // Número de linhas
} Imagem;

// Objetivo: carregar do disco um frame em tons de cinza no formato PGM binário (P5).
// Funcionamento: confere a assinatura "P5", pula espaços e linhas de comentário
// ('#') do cabeçalho, lê largura, altura e valor máximo, e então lê os
// largura*altura bytes de pixels de uma vez para um buffer alocado.
// Encerra o programa se o arquivo não abrir ou não for P5.
// Retorno: struct Imagem com os pixels (o chamador deve liberar imagem.pixels).
Imagem ler_pgm(const char *caminho) {
    FILE *arquivo = fopen(caminho, "rb"); // Abre o arquivo em modo binário (leitura)
    if (!arquivo) {
        fprintf(stderr, "Erro ao abrir %s\n", caminho);
        exit(1);
    }

    char assinatura[3]; // Guarda os 2 caracteres lidos (esperado: P5)
    if (fscanf(arquivo, "%2s", assinatura) != 1 || strcmp(assinatura, "P5") != 0) {
        fprintf(stderr, "Formato inválido (esperado P5)\n"); // Se não for P5, dá erro
        exit(1);
    }

    int largura; // Largura lida do cabeçalho
    int altura; // Altura lida do cabeçalho
    int valor_maximo; // Maior intensidade possível, lida do cabeçalho
    // Tratamento de comentários (pula espaços em branco e linhas '#'
    // até achar o primeiro caractere que não é nem espaço nem comentário)
    int caractere; // Caractere lido do arquivo (int para comportar EOF)
    while (1) {
        caractere = fgetc(arquivo);
        if (caractere == EOF) {
            fprintf(stderr, "Cabeçalho incompleto em %s\n", caminho);
            exit(1);
        }
        if (caractere == '#') {
            while ((caractere = fgetc(arquivo)) != '\n' && caractere != EOF) {
                // Descarta o resto da linha de comentário
            }
        } else if (caractere != ' ' && caractere != '\t' && caractere != '\n' && caractere != '\r') {
            break;
        }
    }
    ungetc(caractere, arquivo); // O laço já leu o 1º dígito da largura; devolve-o ao arquivo para o fscanf ler o número inteiro

    if (fscanf(arquivo, "%d %d %d", &largura, &altura, &valor_maximo) != 3) { // Leitura do cabeçalho
        fprintf(stderr, "Cabeçalho inválido em %s\n", caminho);
        exit(1);
    }
    fgetc(arquivo); // Consome o \n após o cabeçalho; senão o fread o leria como primeiro pixel e deslocaria a imagem

    // Só PGM de 8 bits (1 byte por pixel) com valor máximo 255 (o filtro assume 0..255)
    if (largura <= 0 || altura <= 0 || valor_maximo != 255) {
        fprintf(stderr, "PGM não suportado em %s: dimensões positivas e valor máximo 255 são exigidos\n", caminho);
        exit(1);
    }

    unsigned char *pixels = malloc((size_t)largura * altura); // Alocação de um buffer de largura x altura bytes
    if (!pixels) {
        fprintf(stderr, "Memória insuficiente para %s\n", caminho);
        exit(1);
    }
    // fread devolve quantos bytes conseguiu ler; se for menos que largura * altura,
    // o arquivo acabou antes de todos os pixels (arquivo truncado)
    if (fread(pixels, 1, (size_t)largura * altura, arquivo) != (size_t)largura * altura) {
        fprintf(stderr, "Arquivo %s incompleto (faltam pixels)\n", caminho);
        exit(1);
    }
    fclose(arquivo);

    Imagem imagem = {pixels, largura, altura};
    return imagem;
}

// Objetivo: gravar no disco a imagem borrada.
// Funcionamento: escreve o cabeçalho de um PGM binário (P5) e, em seguida,
// o buffer de pixels inteiro (1 byte por pixel), linha a linha.
// (Equivalente de escrever_ppm do Horn-Schunck: lá são 3 bytes por pixel, aqui 1.)
void escrever_pgm(const char *caminho, const unsigned char *pixels, int largura, int altura) {
    FILE *arquivo = fopen(caminho, "wb"); // Abre arquivo em modo binário (escrita)
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar %s\n", caminho);
        exit(1);
    }
    fprintf(arquivo, "P5\n%d %d\n255\n", largura, altura); // Escreve o cabeçalho P5 (largura, altura, maxval=255)
    fwrite(pixels, 1, (size_t)largura * altura, arquivo); // Escreve o buffer (1 byte por pixel) no arquivo
    fclose(arquivo);
}

// Objetivo: resumo de 32 bits (FNV-1a) do resultado, para comparar rapidamente
// a versão sequencial com a paralela sem trocar arquivos.
// Funcionamento: continua o hash a partir de 'hash' (permite encadear vários frames).
uint32_t atualizar_checksum(uint32_t hash, const unsigned char *pixels, size_t total_pixels) {
    for (size_t posicao = 0; posicao < total_pixels; posicao++) {
        hash ^= pixels[posicao];
        hash *= 16777619u;
    }
    return hash;
}

// Objetivo: calcular UMA iteração do blur 3x3 de forma direta (referência).
// Funcionamento: para cada pixel interno, soma os 9 vizinhos com dois laços
// e grava (soma + 4) / 9 no buffer de destino. A borda não é escrita
// (ambos os buffers já a contêm com o valor original).
// Parâmetros: origem (lida), destino (escrito), largura e altura da imagem.
void filtro_basico_uma_iteracao(const unsigned char *origem, unsigned char *destino, int largura, int altura) {
    if (altura < 3 || largura < 3) {
        return; // Não há pixel interno: nada a fazer
    }

    for (int linha = 1; linha < altura - 1; linha++) { // Linha do pixel atual
        for (int coluna = 1; coluna < largura - 1; coluna++) { // Coluna do pixel atual
            int soma_vizinhanca = 0; // Soma dos 9 pixels da vizinhança 3x3
            for (int linha_vizinha = linha - 1; linha_vizinha <= linha + 1; linha_vizinha++) {
                for (int coluna_vizinha = coluna - 1; coluna_vizinha <= coluna + 1; coluna_vizinha++) {
                    soma_vizinhanca += origem[linha_vizinha * largura + coluna_vizinha];
                }
            }
            destino[linha * largura + coluna] = (unsigned char)((soma_vizinhanca + 4) / 9);
        }
    }
}

// Objetivo: calcular UMA iteração do blur 3x3 com resultado IDÊNTICO ao da
// versão básica, porém mais rápido.
// Funcionamento: percorre cada linha com três ponteiros (linha acima, atual
// e abaixo) e reaproveita as somas de coluna: ao avançar um pixel, só a
// soma da nova coluna é calculada; as duas anteriores são reutilizadas.
// Reduz de 9 para 3 leituras por pixel.
void filtro_otimizado_uma_iteracao(const unsigned char *origem, unsigned char *destino, int largura, int altura) {
    if (altura < 3 || largura < 3) {
        return;
    }

    for (int linha = 1; linha < altura - 1; linha++) { // Linha do pixel atual
        const unsigned char *linha_acima = origem + (size_t)(linha - 1) * largura;
        const unsigned char *linha_atual = origem + (size_t)linha * largura;
        const unsigned char *linha_abaixo = origem + (size_t)(linha + 1) * largura;
        unsigned char *linha_destino = destino + (size_t)linha * largura;

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

// Objetivo: rodar as N iterações do blur sobre a imagem inteira (equivalente de
// executar_jacobi do Horn-Schunck).
// Funcionamento: em cada iteração, lê de _atual e escreve em _novo; depois
// troca os ponteiros. PRÉ-CONDIÇÃO: os dois buffers começam com o mesmo
// conteúdo (a borda nunca é reescrita).
// Parâmetros: os dois buffers são passados por ponteiro (unsigned char **)
// porque as trocas mudam para onde cada um aponta, e quem chama precisa saber
// em qual buffer ficou o resultado final (_atual).
void executar_blur(
    unsigned char **ponteiro_atual,
    unsigned char **ponteiro_novo,
    int largura,
    int altura,
    int num_iteracoes,
    int modo) {
    // Cópias locais dos ponteiros, pra não escrever *ponteiro_... em todo o loop
    unsigned char *buffer_atual = *ponteiro_atual; // Imagem da iteração anterior (lida)
    unsigned char *buffer_novo = *ponteiro_novo; // Imagem sendo calculada (escrita)

    for (int iteracao = 0; iteracao < num_iteracoes; iteracao++) { // Iteração atual do blur
        if (modo == MODO_OTIMIZADO) {
            filtro_otimizado_uma_iteracao(buffer_atual, buffer_novo, largura, altura);
        } else {
            filtro_basico_uma_iteracao(buffer_atual, buffer_novo, largura, altura);
        }

        // Troca de buffers (swap de ponteiros): _atual passa a apontar para o
        // resultado recém-calculado (lido na próxima iteração) e _novo para o
        // buffer antigo (sobrescrito na próxima iteração)
        unsigned char *ponteiro_temporario = buffer_atual; // Guarda o endereço de _atual, que se perderia ao sobrescrevê-lo
        buffer_atual = buffer_novo;
        buffer_novo = ponteiro_temporario;
    }

    // Devolve a quem chamou para onde cada ponteiro aponta depois das trocas
    *ponteiro_atual = buffer_atual;
    *ponteiro_novo = buffer_novo;
}

// Área de trabalho do cálculo de um frame: alocada uma vez e reaproveitada para todos os frames
typedef struct {
    // _atual/_novo em pares porque o blur não pode escrever a nova iteração em cima da que ainda está sendo lida
    unsigned char *buffer_atual; // Imagem da iteração anterior (no fim: o resultado)
    unsigned char *buffer_novo; // Imagem sendo calculada
} AreaTrabalho;

// Objetivo: calcular o blur iterado de um frame (equivalente de calcular_par).
// Funcionamento:
//   1) copia o frame para os DOIS buffers (assim a borda, que nunca é
//      processada, fica correta mesmo após as trocas de ponteiros);
//   2) repete N iterações do blur (executar_blur).
// O resultado fica em area->buffer_atual.
void calcular_frame(Imagem *frame, AreaTrabalho *area, int num_iteracoes, int modo) {
    size_t total_pixels = (size_t)frame->largura * frame->altura; // Total de pixels

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

// Objetivo: instante atual em segundos (relógio monotônico), para cronometrar.
double agora(void) {
    struct timespec instante; // Segundos + nanossegundos
    clock_gettime(CLOCK_MONOTONIC, &instante);
    return instante.tv_sec + instante.tv_nsec / 1e9;
}

// Objetivo: aplicar o blur 3x3 iterado a uma sequência de frames, de forma
// sequencial: um frame por vez, gerando uma imagem borrada por frame.
// Funcionamento:
//   1) lê as opções e os argumentos;
//   2) para cada frame i: lê o frame, calcula o blur (calcular_frame) e grava
//      pasta_saida/blur_<i>.pgm;
//   3) mede o tempo total (leitura + cálculo + escrita), o mesmo que o paralelo mede.
// Uso: [-b] iteracoes pasta_saida frame0.pgm frame1.pgm ... frameN-1.pgm
int main(int argc, char **argv) {
    int modo = MODO_OTIMIZADO; // -b: usa o filtro básico
    int opcao; // Letra da opção lida pelo getopt
    int opcoes_validas = 1; // Vira 0 se aparecer uma opção desconhecida
    while ((opcao = getopt(argc, argv, "b")) != -1) {
        if (opcao == 'b') {
            modo = MODO_BASICO;
        } else {
            opcoes_validas = 0;
        }
    }
    int num_frames = argc - optind - 2; // Depois de iteracoes e pasta_saida vêm os frames
    if (!opcoes_validas || num_frames < 1) {
        fprintf(stderr, "Uso: %s [-b] iteracoes pasta_saida frame0.pgm frame1.pgm ...\n", argv[0]);
        return 1;
    }

    int num_iteracoes = atoi(argv[optind]); // N: número de iterações do blur
    const char *pasta_saida = argv[optind + 1]; // Onde gravar blur_0000.pgm, blur_0001.pgm, ...
    char **caminhos_frames = argv + optind + 2; // Arquivos PGM dos frames, em ordem

    // Valida os parâmetros numéricos (atoi devolve 0 para texto inválido)
    if (num_iteracoes <= 0) {
        fprintf(stderr, "iteracoes precisa ser um inteiro positivo\n");
        return 1;
    }
    if (CRIAR_PASTA(pasta_saida) != 0 && errno != EEXIST) {
        fprintf(stderr, "Erro ao criar a pasta %s\n", pasta_saida);
        return 1;
    }

    // Cronometra tudo (leitura + cálculo + escrita), separando o tempo de cada etapa
    double tempo_leitura = 0.0; // Soma do tempo lendo frames
    double tempo_calculo = 0.0; // Soma do tempo calculando (blur)
    double tempo_escrita = 0.0; // Soma do tempo gravando resultados
    double instante_inicio = agora();

    double inicio = agora(); // Início da etapa sendo cronometrada
    Imagem frame = ler_pgm(caminhos_frames[0]); // Primeiro frame (define as dimensões)
    tempo_leitura += agora() - inicio;
    int largura = frame.largura; // Número de colunas da imagem
    int altura = frame.altura; // Número de linhas da imagem
    size_t total_pixels = (size_t)largura * altura; // Total de pixels

    AreaTrabalho area; // Reaproveitada para todos os frames
    area.buffer_atual = malloc(total_pixels);
    area.buffer_novo = malloc(total_pixels);
    if (!area.buffer_atual || !area.buffer_novo) {
        fprintf(stderr, "Memória insuficiente para os buffers de trabalho\n");
        return 1;
    }

    uint32_t checksum = 2166136261u; // Resumo de todos os resultados, em ordem

    for (int indice = 0; indice < num_frames; indice++) { // Frame atual
        // Leitura (o primeiro frame já foi lido acima, para obter as dimensões)
        if (indice > 0) {
            inicio = agora();
            frame = ler_pgm(caminhos_frames[indice]);
            tempo_leitura += agora() - inicio;
            // Confere se tem a mesma dimensão do primeiro frame (buffers de trabalho reaproveitados)
            if (frame.largura != largura || frame.altura != altura) {
                fprintf(stderr, "%s tem dimensões diferentes do primeiro frame\n", caminhos_frames[indice]);
                return 1;
            }
        }

        // Cálculo
        inicio = agora();
        calcular_frame(&frame, &area, num_iteracoes, modo);
        tempo_calculo += agora() - inicio;

        // Escrita
        inicio = agora();
        char caminho[4096]; // Caminho do arquivo de saída
        snprintf(caminho, sizeof(caminho), "%s/blur_%04d.pgm", pasta_saida, indice);
        escrever_pgm(caminho, area.buffer_atual, largura, altura);
        tempo_escrita += agora() - inicio;

        checksum = atualizar_checksum(checksum, area.buffer_atual, total_pixels);
        free(frame.pixels);
    }

    double tempo_total = agora() - instante_inicio; // Em segundos
    printf("Frames: %d de %dx%d pixels\n", num_frames, largura, altura);
    printf("Vídeo processado em %.6f s (%d frames, sequencial, modo %s, %d iterações; %.2f frames/s)\n",
        tempo_total, num_frames, modo == MODO_OTIMIZADO ? "otimizado" : "básico", num_iteracoes, num_frames / tempo_total);
    printf("Tempo por etapa: leitura %.3f s, cálculo %.3f s, escrita %.3f s\n",
        tempo_leitura, tempo_calculo, tempo_escrita);
    printf("Checksum do resultado: %08x\n", (unsigned)checksum);
    printf("Imagens salvas em %s/blur_*.pgm\n", pasta_saida);

    free(area.buffer_atual);
    free(area.buffer_novo);

    return 0;
}
