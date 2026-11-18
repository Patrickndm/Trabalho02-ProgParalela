# Trabalho 02 - Blur 3x3 iterado com MPI

UEM - Ciência da Computação - Programação Paralela (DIN12041)
Prof. Ronaldo Augusto de Lara Goncalves

## Integrantes

- Matheus Foltran Consonni - RA 133862
- Patrick Nelson de Morais - RA 130166

## O problema

Aplicar N vezes um filtro de desfoque (média 3x3) em uma sequência de frames
em tons de cinza (PGM, 1 byte por pixel).

- Pixel interno = `(soma dos 9 vizinhos + 4) / 9` (aritmética inteira).
- Pixels da borda da imagem não mudam.
- Cada iteração lê de uma imagem e escreve em OUTRA (dois buffers).

## Arquivos

| Arquivo | Descrição |
| ------- | --------- |
| `sequencial.c` | Versão sequencial (baseline e gabarito de corretude) |
| `paralelo.c` | Versão paralela com MPI |
| `Makefile` | Compila as duas versões |
| `benchmark/gerar_frames.py` | Gera os frames sintéticos em várias resoluções |
| `benchmark/executar_benchmark.sh` | Teste de corretude + medição de tempos |
| `benchmark/gerar_graficos.py` | Gráficos (PDF/PNG) e `resumo.txt` |

## Instalação (Debian/Ubuntu/WSL)

```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-numpy python3-matplotlib
```

## Compilação

```bash
make            # gera blur_seq e blur_par
```

ou manualmente:

```bash
gcc  -O2 -o blur_seq sequencial.c
mpicc -O2 -o blur_par paralelo.c
```

## Execução

```bash
# sequencial: iteracoes pasta_saida frames...
./blur_seq 100 saida_seq frame1.pgm frame2.pgm

# paralelo: P processos, entrada, número de iterações e frames são parâmetros
mpirun -np 4 ./blur_par 100 saida_par frame1.pgm frame2.pgm
```

Opções do paralelo:

| Opção | Significado | Padrão |
| ----- | ----------- | ------ |
| `-np P` (do `mpirun`) | número de processos | - |
| `-b B` | iterações por rodada = linhas de halo (1 = trocar a cada iteração) | 4 |
| `-v PASTA` | teste de corretude: compara a saída com a do sequencial em `PASTA` | desligado |

Se pedir mais processos do que núcleos, use `mpirun --oversubscribe -np P ...`.

Sequencial: `-b` usa o filtro básico (9 leituras por pixel) em vez do otimizado.

## Teste de corretude (só roda quando você pedir)

```bash
./blur_seq 100 ref frame1.pgm frame2.pgm
mpirun -np 4 ./blur_par -v ref 100 saida_par frame1.pgm frame2.pgm
```

Depois de medir o tempo, o processo 0 compara cada `blur_NNNN.pgm` gerado com o
do sequencial (pixel a pixel) e imprime quantos pixels diferem. O código de
saída é 1 se houver diferença. Também dá para comparar com `cmp` e pelo
`Checksum do resultado`, que é igual nas duas versões.

## Como o paralelo funciona

- A imagem é dividida em faixas de linhas, uma por processo (decomposição de domínio).
- O processo 0 lê o frame, envia as faixas (`MPI_Send`/`MPI_Recv`), calcula a
  sua, recebe as faixas prontas e grava a imagem.
- Cada processo guarda também linhas de **halo** (cópias das linhas dos
  vizinhos). Os halos são trocados no início de cada rodada de B iterações
  (`MPI_Send`/`MPI_Recv`, ordem escolhida para não haver deadlock).
- **Bloqueio temporal**: dentro de uma rodada, cada iteração calcula 1 linha a
  menos em cada ponta do halo (trabalho redundante), e assim B iterações
  são feitas com uma única troca de mensagens.
- Sincronização: `MPI_Bcast` (dimensões e aviso de erro), `MPI_Barrier`
  (início e fim do cronômetro) e a troca de halos entre vizinhos.
- Resultado idêntico bit a bit ao do sequencial.

## Avaliação de desempenho

```bash
./benchmark/executar_benchmark.sh      # gera frames, testa corretude, mede tempos
python3 benchmark/gerar_graficos.py    # gera graficos/ e resumo.txt
```

Os parâmetros mudam por variável de ambiente, por exemplo:

```bash
RESOLUCOES="640x480" PROCESSOS="1 2 4" REPETICOES=1 ./benchmark/executar_benchmark.sh
```

Resultados em `benchmark/resultados/` (`tempos.csv`, `corretude.csv`,
`ambiente.txt`, `resumo.txt`, `graficos/`).
