import cv2
import numpy as np


def preprocessar(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray


def segmentar(img):

    gray = preprocessar(img)

    _, escuro = cv2.threshold(
        gray,
        85,
        255,
        cv2.THRESH_BINARY_INV
    )

    bordas = cv2.Canny(
        gray,
        30,
        120
    )

    kernel = np.ones((5, 5), np.uint8)

    bordas = cv2.dilate(
        bordas,
        kernel,
        iterations=2
    )

    mascara = cv2.bitwise_or(
        escuro,
        bordas
    )

    kernel = np.ones((11, 11), np.uint8)

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8)
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = np.zeros(gray.shape, np.uint8)

    if len(contornos) == 0:
        return resultado

    maior = max(contornos, key=cv2.contourArea)

    cv2.drawContours(
        resultado,
        [maior],
        -1,
        255,
        -1
    )

    return resultado


def extrair_caracteristicas(mascara):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        return None

    cnt = max(contornos, key=cv2.contourArea)

    area = cv2.contourArea(cnt)

    perimetro = cv2.arcLength(
        cnt,
        True
    )

    rect = cv2.minAreaRect(cnt)

    w, h = rect[1]

    if w < h:
        w, h = h, w

    aspect = w / h if h else 0

    rect_area = w * h

    extent = area / rect_area if rect_area else 0

    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area else 0

    circularidade = (
        4 * np.pi * area / (perimetro ** 2)
        if perimetro else 0
    )

    hu = cv2.HuMoments(
        cv2.moments(cnt)
    ).flatten()

    hu = np.array([
        -np.sign(v) * np.log10(abs(v) + 1e-10)
        for v in hu
    ])

    epsilon = 0.01 * perimetro

    aprox = cv2.approxPolyDP(
        cnt,
        epsilon,
        True
    )

    num_vertices = len(aprox)

    hull_indices = cv2.convexHull(
        cnt,
        returnPoints=False
    )

    defeitos = 0

    if hull_indices is not None and len(hull_indices) > 3:

        d = cv2.convexityDefects(
            cnt,
            hull_indices
        )

        if d is not None:
            defeitos = len(d)

    caracteristicas = [

        area,
        perimetro,

        w,
        h,

        aspect,

        extent,

        solidity,

        circularidade,

        *hu,

        hull_area,

        num_vertices,

        defeitos
    ]

    return caracteristicas


def desenhar(img, mascara):

    resultado = img.copy()

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos):

        maior = max(
            contornos,
            key=cv2.contourArea
        )

        cv2.drawContours(
            resultado,
            [maior],
            -1,
            (0, 255, 0),
            3
        )

    return resultado