#!/usr/bin/env python3
# Gera os gráficos (PDF e PNG) e o resumo em texto a partir de resultados/tempos.csv.
# Uso: python3 gerar_graficos.py [--resultados PASTA] [--b-padrao 4]
# Requer: numpy e matplotlib.

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ler_tempos(caminho):
    """Lê tempos.csv e devolve {(programa, resolucao, processos, bloco): [tempos]}."""
    dados = defaultdict(list)
    larguras = {}
    with open(caminho, newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            resolucao = linha["resolucao"]
            larguras[resolucao] = int(linha["largura"]) * int(linha["altura"])
            chave = (linha["programa"], resolucao, int(linha["processos"]), int(linha["bloco"]))
            dados[chave].append(float(linha["tempo_s"]))
    return dados, larguras


def salvar(figura, pasta, nome):
    figura.tight_layout()
    figura.savefig(os.path.join(pasta, nome + ".pdf"))
    figura.savefig(os.path.join(pasta, nome + ".png"), dpi=150)
    plt.close(figura)


def main():
    analisador = argparse.ArgumentParser()
    analisador.add_argument("--resultados", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"))
    analisador.add_argument("--b-padrao", type=int, default=4)
    argumentos = analisador.parse_args()

    dados, pixels = ler_tempos(os.path.join(argumentos.resultados, "tempos.csv"))
    pasta_graficos = os.path.join(argumentos.resultados, "graficos")
    os.makedirs(pasta_graficos, exist_ok=True)

    resolucoes = sorted(pixels, key=lambda r: pixels[r])
    todos_processos = sorted({chave[2] for chave in dados if chave[0] == "paralelo"})
    p_max = max(todos_processos)

    def media(chave):
        return float(np.mean(dados[chave]))

    def desvio(chave):
        return float(np.std(dados[chave], ddof=1)) if len(dados[chave]) > 1 else 0.0

    linhas_resumo = []

    # ---- Escalabilidade: tempo, speedup e eficiência x processos (B = b_padrao) ----
    series = {}
    for resolucao in resolucoes:
        tempo_seq = media(("sequencial", resolucao, 1, 0))
        processos_lista = [p for p in todos_processos if ("paralelo", resolucao, p, argumentos.b_padrao) in dados]
        tempos = [media(("paralelo", resolucao, p, argumentos.b_padrao)) for p in processos_lista]
        desvios = [desvio(("paralelo", resolucao, p, argumentos.b_padrao)) for p in processos_lista]
        series[resolucao] = (tempo_seq, processos_lista, tempos, desvios)
        for p, t, d in zip(processos_lista, tempos, desvios):
            linhas_resumo.append(
                f"{resolucao:>10}  P={p:<3} B={argumentos.b_padrao:<2} tempo {t:8.3f} s (±{d:.3f})  "
                f"speedup {tempo_seq / t:6.2f}  eficiência {tempo_seq / t / p * 100:6.1f}%")
        linhas_resumo.append(f"{resolucao:>10}  sequencial: {tempo_seq:.3f} s (±{desvio(('sequencial', resolucao, 1, 0)):.3f})")
        linhas_resumo.append("")

    figura, eixo = plt.subplots(figsize=(7, 4.5))
    for resolucao, (tempo_seq, processos_lista, tempos, desvios) in series.items():
        linha = eixo.errorbar(processos_lista, tempos, yerr=desvios, marker="o", capsize=3, label=resolucao)
        eixo.axhline(tempo_seq, color=linha[0].get_color(), linestyle="--", linewidth=0.9, alpha=0.7)
    eixo.set_xlabel("Processos MPI")
    eixo.set_ylabel("Tempo total (s)")
    eixo.set_yscale("log")
    eixo.set_title("Tempo de execução (tracejado: sequencial)")
    eixo.grid(True, alpha=0.3)
    eixo.legend(title="Resolução")
    salvar(figura, pasta_graficos, "tempo")

    figura, eixo = plt.subplots(figsize=(7, 4.5))
    for resolucao, (tempo_seq, processos_lista, tempos, _) in series.items():
        eixo.plot(processos_lista, [tempo_seq / t for t in tempos], marker="o", label=resolucao)
    eixo.plot(todos_processos, todos_processos, "k--", linewidth=0.9, label="ideal")
    eixo.set_xlabel("Processos MPI")
    eixo.set_ylabel("Speedup (T_seq / T_par)")
    eixo.set_title("Speedup")
    eixo.grid(True, alpha=0.3)
    eixo.legend(title="Resolução")
    salvar(figura, pasta_graficos, "speedup")

    figura, eixo = plt.subplots(figsize=(7, 4.5))
    for resolucao, (tempo_seq, processos_lista, tempos, _) in series.items():
        eixo.plot(processos_lista, [tempo_seq / t / p * 100 for p, t in zip(processos_lista, tempos)], marker="o", label=resolucao)
    eixo.axhline(100, color="k", linestyle="--", linewidth=0.9)
    eixo.set_xlabel("Processos MPI")
    eixo.set_ylabel("Eficiência (%)")
    eixo.set_title("Eficiência (speedup / processos)")
    eixo.grid(True, alpha=0.3)
    eixo.legend(title="Resolução")
    salvar(figura, pasta_graficos, "eficiencia")

    # ---- Bloqueio temporal: P = p_max, variando B ----
    figura, eixo = plt.subplots(figsize=(7, 4.5))
    linhas_resumo.append(f"Bloqueio temporal (P = {p_max}):")
    for resolucao in resolucoes:
        tempo_seq = media(("sequencial", resolucao, 1, 0))
        blocos = sorted({chave[3] for chave in dados if chave[0] == "paralelo" and chave[1] == resolucao and chave[2] == p_max})
        if len(blocos) < 2:
            continue
        speedups = [tempo_seq / media(("paralelo", resolucao, p_max, b)) for b in blocos]
        eixo.plot(blocos, speedups, marker="o", label=resolucao)
        for b, s in zip(blocos, speedups):
            linhas_resumo.append(f"{resolucao:>10}  B={b:<3} speedup {s:6.2f}")
    eixo.set_xscale("log", base=2)
    eixo.set_xlabel("Iterações por rodada (B = linhas de halo)")
    eixo.set_ylabel("Speedup")
    eixo.set_title(f"Efeito do bloqueio temporal ({p_max} processos)")
    eixo.grid(True, alpha=0.3)
    eixo.legend(title="Resolução")
    salvar(figura, pasta_graficos, "bloco")

    # ---- Corretude ----
    caminho_corretude = os.path.join(argumentos.resultados, "corretude.csv")
    if os.path.exists(caminho_corretude):
        with open(caminho_corretude, newline="") as arquivo:
            linhas = list(csv.DictReader(arquivo))
        ok = sum(1 for l in linhas if l["resultado"] == "OK")
        linhas_resumo.append("")
        linhas_resumo.append(f"Corretude: {ok} de {len(linhas)} configurações idênticas ao sequencial")

    with open(os.path.join(argumentos.resultados, "resumo.txt"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas_resumo) + "\n")
    print("\n".join(linhas_resumo))
    print(f"\nGráficos em {pasta_graficos}/")


if __name__ == "__main__":
    main()
