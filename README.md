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

Se pedir mais processos do que núcleos, use `mpirun --oversubscribe -np P ...`.

## Teste de corretude

As imagens geradas pelas duas versões devem ser idênticas:

```bash
./blur_seq 100 saida_seq frame1.pgm frame2.pgm
mpirun -np 4 ./blur_par 100 saida_par frame1.pgm frame2.pgm
diff -r saida_seq saida_par && echo iguais
```

## Como o paralelo funciona

- A imagem é dividida em faixas de linhas, uma por processo (decomposição de domínio).
- O processo 0 lê o frame, envia as faixas (`MPI_Send`/`MPI_Recv`), calcula a
  sua, recebe as faixas prontas e grava a imagem.
- Cada processo guarda também uma linha de **halo** em cima e outra embaixo
  (cópias da última linha do vizinho de cima e da primeira do vizinho de
  baixo). Antes de cada iteração os vizinhos trocam essas linhas
  (`MPI_Send`/`MPI_Recv`, em uma ordem que evita deadlock).
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
