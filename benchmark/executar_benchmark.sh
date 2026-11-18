#!/usr/bin/env bash
# Avaliação de desempenho do blur 3x3 iterado: sequencial x paralelo (MPI).
#
# Etapas:
#   1) compila (make) e gera as sequências de frames, se ainda não existirem;
#   2) grava as informações da máquina em resultados/ambiente.txt;
#   3) teste de corretude: em cada resolução, compara (diff) as imagens do
#      paralelo, com cada número de processos, com as do sequencial
#      -> resultados/corretude.csv;
#   4) medição: para cada resolução, uma execução de aquecimento descartada e
#      depois REPETICOES rodadas. Em cada rodada roda o sequencial e o paralelo
#      com cada número de processos (resultados/tempos.csv). As rodadas são
#      intercaladas para que variações da máquina afetem todas por igual.
#
# O tempo medido é o total do programa (leitura + cálculo + escrita), igual nos dois.
#
# Os parâmetros podem ser trocados por variáveis de ambiente, por exemplo:
#   REPETICOES=1 RESOLUCOES="640x480" PROCESSOS="1 2 4" ./executar_benchmark.sh
#   RESULTADOS=/outra/pasta ./executar_benchmark.sh   (não sobrescreve benchmark/resultados)

set -euo pipefail

RESOLUCOES=${RESOLUCOES:-"640x480 1280x720 1920x1080 3840x2160"}
ITERACOES=${ITERACOES:-"100"}
FRAMES_POR_VIDEO=${FRAMES_POR_VIDEO:-16}
PROCESSOS=${PROCESSOS:-"1 2 4 8 12 16"}
REPETICOES=${REPETICOES:-5}
ITERACOES_CORRETUDE=${ITERACOES_CORRETUDE:-50}

NUCLEOS_LOGICOS=$(nproc)

PASTA_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(dirname "$PASTA_SCRIPT")"
FRAMES="$PASTA_SCRIPT/frames"
RESULTADOS="${RESULTADOS:-$PASTA_SCRIPT/resultados}"
TEMPORARIOS="$(mktemp -d)"
trap 'rm -rf "$TEMPORARIOS"' EXIT

SEQ="$RAIZ/blur_seq"
PAR="$RAIZ/blur_par"

# Executa o paralelo com P processos (--oversubscribe permite P maior que os núcleos)
executar_paralelo() {
    local processos=$1; shift
    mpirun --oversubscribe -np "$processos" "$PAR" "$@"
}

# Extrai o tempo da linha "Vídeo processado em X s (...)"
extrair_tempo() {
    sed -n 's/.*processado em \([0-9.]*\) s.*/\1/p'
}

# Frames da sequência de uma resolução, em ordem
frames_video() {
    ls "$FRAMES/video_$1"/frame_*.pgm | head -n "$FRAMES_POR_VIDEO"
}

# ---------- 1) Compilação e frames ----------
echo "== Compilando"
make -C "$RAIZ" -s

for resolucao in $RESOLUCOES; do
    if (( $(ls "$FRAMES/video_$resolucao"/frame_*.pgm 2>/dev/null | wc -l) < FRAMES_POR_VIDEO )); then
        python3 "$PASTA_SCRIPT/gerar_frames.py" --video "$FRAMES_POR_VIDEO" "$FRAMES" "$resolucao"
    fi
done

mkdir -p "$RESULTADOS"

# ---------- 2) Ambiente ----------
echo "== Registrando o ambiente em $RESULTADOS/ambiente.txt"
{
    echo "Data: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Sistema: $(uname -srm)"
    [[ -f /etc/os-release ]] && grep PRETTY_NAME /etc/os-release
    grep -qi microsoft /proc/version 2>/dev/null && echo "WSL: sim ($(cat /proc/version))"
    echo "Processadores lógicos (nproc): $NUCLEOS_LOGICOS"
    echo
    lscpu
    echo
    free -h
    echo
    gcc --version | head -1
    mpirun --version | head -1
    echo "Flags: $(sed -n 's/^CFLAGS *= *//p' "$RAIZ/Makefile")"
    echo
    echo "Parâmetros: RESOLUCOES=\"$RESOLUCOES\" ITERACOES=\"$ITERACOES\" FRAMES_POR_VIDEO=$FRAMES_POR_VIDEO REPETICOES=$REPETICOES PROCESSOS=\"$PROCESSOS\""
} > "$RESULTADOS/ambiente.txt"

# ---------- 3) Corretude ----------
echo "== Teste de corretude (K=$ITERACOES_CORRETUDE): compara as imagens do paralelo com as do sequencial"
echo "resolucao,processos,resultado" > "$RESULTADOS/corretude.csv"
for resolucao in $RESOLUCOES; do
    mapfile -t frames < <(frames_video "$resolucao")
    rm -rf "$TEMPORARIOS/sequencial"
    "$SEQ" "$ITERACOES_CORRETUDE" "$TEMPORARIOS/sequencial" "${frames[@]}" > /dev/null
    for processos in $PROCESSOS; do
        rm -rf "$TEMPORARIOS/paralelo"
        executar_paralelo "$processos" "$ITERACOES_CORRETUDE" "$TEMPORARIOS/paralelo" "${frames[@]}" > /dev/null
        if diff -r -q "$TEMPORARIOS/sequencial" "$TEMPORARIOS/paralelo" > /dev/null; then
            resultado=OK
        else
            resultado=FALHOU
        fi
        echo "$resolucao,$processos,$resultado" >> "$RESULTADOS/corretude.csv"
        printf '  %-10s %2s processos: %s\n' "$resolucao" "$processos" "$resultado"
    done
done

# ---------- 4) Medição ----------
total=0
for resolucao in $RESOLUCOES; do for k in $ITERACOES; do
    total=$(( total + REPETICOES * (1 + $(wc -w <<< "$PROCESSOS")) ))
done; done

echo "== Medição de tempos ($total execuções; processos: $PROCESSOS)"
arquivo_csv="$RESULTADOS/tempos.csv"
echo "programa,resolucao,largura,altura,iteracoes,frames,processos,repeticao,tempo_s" > "$arquivo_csv"
feitas=0
inicio_geral=$(date +%s)

for resolucao in $RESOLUCOES; do
    largura=${resolucao%x*}
    altura=${resolucao#*x}
    mapfile -t frames < <(frames_video "$resolucao")
    num_frames=${#frames[@]}

    for k in $ITERACOES; do
        # Aquecimento (descartado): carrega os arquivos no cache do sistema e
        # tira o processador do estado de economia de energia
        "$SEQ" "$k" "$TEMPORARIOS/saida" "${frames[@]}" > /dev/null
        executar_paralelo "$NUCLEOS_LOGICOS" "$k" "$TEMPORARIOS/saida" "${frames[@]}" > /dev/null

        for repeticao in $(seq 1 "$REPETICOES"); do
            rm -rf "$TEMPORARIOS/saida"
            tempo=$("$SEQ" "$k" "$TEMPORARIOS/saida" "${frames[@]}" | extrair_tempo)
            echo "sequencial,$resolucao,$largura,$altura,$k,$num_frames,1,$repeticao,$tempo" >> "$arquivo_csv"
            feitas=$(( feitas + 1 ))

            for processos in $PROCESSOS; do
                rm -rf "$TEMPORARIOS/saida"
                tempo=$(executar_paralelo "$processos" "$k" "$TEMPORARIOS/saida" "${frames[@]}" | extrair_tempo)
                echo "paralelo,$resolucao,$largura,$altura,$k,$num_frames,$processos,$repeticao,$tempo" >> "$arquivo_csv"
                feitas=$(( feitas + 1 ))
            done

            decorrido=$(( $(date +%s) - inicio_geral ))
            printf '  %-10s K=%-4s rodada %s/%s  (%d/%d execuções, %d s decorridos)\n' \
                "$resolucao" "$k" "$repeticao" "$REPETICOES" "$feitas" "$total" "$decorrido"
        done
    done
done

echo "== Concluído em $(( $(date +%s) - inicio_geral )) s"
echo "Resultados em $RESULTADOS/"
echo "Para gerar os gráficos: python3 $PASTA_SCRIPT/gerar_graficos.py"
