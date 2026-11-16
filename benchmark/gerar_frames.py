#!/usr/bin/env python3
# Gera os frames sintéticos (PGM) usados no benchmark.
# Uso: python3 gerar_frames.py [--video N] [pasta_saida] [LARGURAxALTURA ...]

import argparse
import math
import os

import numpy as np

RESOLUCOES_PADRAO = ["640x480", "1280x720", "1920x1080", "3840x2160"]
DESLOCAMENTO_FUNDO = (1.0, 0.5)
DESLOCAMENTO_OBJETO = (-1.0, 1.0)
SEMENTE = 2026
MARGEM = 16


def textura(altura, largura, escala, gerador):
    ruido = gerador.standard_normal((altura, largura))
    freq_y = np.fft.fftfreq(altura)[:, None]
    freq_x = np.fft.fftfreq(largura)[None, :]
    filtro = np.exp(-2 * (np.pi * escala) ** 2 * (freq_x ** 2 + freq_y ** 2))
    return np.fft.ifft2(np.fft.fft2(ruido) * filtro).real


def deslocar(imagem, dx, dy):
    altura, largura = imagem.shape
    freq_y = np.fft.fftfreq(altura)[:, None]
    freq_x = np.fft.fftfreq(largura)[None, :]
    fase = np.exp(-2j * np.pi * (freq_x * dx + freq_y * dy))
    return np.fft.ifft2(np.fft.fft2(imagem) * fase).real


def normalizar(imagem):
    return (imagem - imagem.min()) / (imagem.max() - imagem.min())


def mascara_disco(altura, largura, centro_x, centro_y, raio):
    y, x = np.mgrid[0:altura, 0:largura]
    distancia = np.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
    return np.clip(raio - distancia + 0.5, 0.0, 1.0)


def gerar_sequencia(largura_final, altura_final, num_frames):
    gerador = np.random.default_rng(SEMENTE)
    maior_deslocamento = (num_frames - 1) * max(abs(d) for d in DESLOCAMENTO_FUNDO + DESLOCAMENTO_OBJETO)
    margem = max(MARGEM, math.ceil(maior_deslocamento) + 2)
    largura = largura_final + 2 * margem
    altura = altura_final + 2 * margem
    escala = max(largura_final, altura_final) / 160
    fundo = normalizar(textura(altura, largura, escala, gerador) + 0.5 * textura(altura, largura, escala / 3, gerador))
    objeto = normalizar(textura(altura, largura, escala / 2, gerador))

    raio = min(largura_final, altura_final) / 4
    centro_x, centro_y = largura / 2, altura / 2

    def para_8_bits(imagem):
        return np.clip(np.rint(16 + 223 * imagem), 0, 255).astype(np.uint8)

    recorte = (slice(margem, margem + altura_final), slice(margem, margem + largura_final))
    frames = []
    for i in range(num_frames):
        if i == 0:
            fundo_i, objeto_i = fundo, objeto
        else:
            fundo_i = deslocar(fundo, i * DESLOCAMENTO_FUNDO[0], i * DESLOCAMENTO_FUNDO[1])
            objeto_i = deslocar(objeto, i * DESLOCAMENTO_OBJETO[0], i * DESLOCAMENTO_OBJETO[1])
        mascara = mascara_disco(altura, largura, centro_x + i * DESLOCAMENTO_OBJETO[0],
                                centro_y + i * DESLOCAMENTO_OBJETO[1], raio)
        frames.append(para_8_bits(((1 - mascara) * fundo_i + mascara * objeto_i)[recorte]))
    return frames


def gerar_par(largura_final, altura_final):
    return tuple(gerar_sequencia(largura_final, altura_final, 2))


def escrever_pgm(caminho, imagem):
    altura, largura = imagem.shape
    with open(caminho, "wb") as arquivo:
        arquivo.write(f"P5\n{largura} {altura}\n255\n".encode())
        arquivo.write(imagem.tobytes())


def main():
    analisador = argparse.ArgumentParser(description="Gera frames sintéticos (PGM P5) para a avaliação de desempenho.")
    analisador.add_argument("--video", type=int, metavar="N", help="gera uma sequência de N frames em vez de um par")
    analisador.add_argument("pasta", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames"))
    analisador.add_argument("resolucoes", nargs="*", default=RESOLUCOES_PADRAO, metavar="LARGURAxALTURA")
    argumentos = analisador.parse_args()
    os.makedirs(argumentos.pasta, exist_ok=True)

    for resolucao in argumentos.resolucoes:
        largura, altura = (int(valor) for valor in resolucao.lower().split("x"))
        if argumentos.video:
            pasta_video = os.path.join(argumentos.pasta, f"video_{resolucao}")
            os.makedirs(pasta_video, exist_ok=True)
            for i, frame in enumerate(gerar_sequencia(largura, altura, argumentos.video)):
                escrever_pgm(os.path.join(pasta_video, f"frame_{i:04d}.pgm"), frame)
            print(f"{resolucao}: {argumentos.video} frames em {pasta_video}/")
        else:
            frame_a, frame_b = gerar_par(largura, altura)
            escrever_pgm(os.path.join(argumentos.pasta, f"{resolucao}_1.pgm"), frame_a)
            escrever_pgm(os.path.join(argumentos.pasta, f"{resolucao}_2.pgm"), frame_b)
            print(f"{resolucao}: {argumentos.pasta}/{resolucao}_1.pgm e {resolucao}_2.pgm")


if __name__ == "__main__":
    main()
