import cv2
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from funcoes import segmentar, extrair_caracteristicas

BASE_DIR = Path(__file__).resolve().parent.parent

modelo = joblib.load(BASE_DIR / "modelos" / "modelo.pkl")

pasta = BASE_DIR / "dataset" / "multiplos"

imagens = sorted(pasta.glob("*.jpg"))
imagens += sorted(pasta.glob("*.jpeg"))
imagens += sorted(pasta.glob("*.png"))

if len(imagens) == 0:
    print("Nenhuma imagem encontrada.")
    exit()

def resize_with_padding(img, size=500):
    h, w = img.shape[:2]
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if len(img.shape) == 2:
        canvas = np.zeros((size, size), dtype=np.uint8)
    else:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return canvas

total_geral_moedas = 0
total_geral_lapis = 0

indice = 1

for arquivo in imagens:
    print("===================")
    print(f"Imagem {indice}/{len(imagens)} {arquivo.name}")
    print("===================")

    imagem = cv2.imread(str(arquivo))
    
    if imagem is None:
        continue

    imagem = imagem[::4, ::4]
    resultado = imagem.copy()

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # OTSU
    _, mascara = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    moedas = 0
    lapis = 0

    print(f"Contornos encontrados: {len(contornos)}")

    for cnt in contornos:
        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        margem = 20
        x = max(0, x - margem)
        y = max(0, y - margem)

        w = min(imagem.shape[1] - x, w + 2 * margem)
        h = min(imagem.shape[0] - y, h + 2 * margem)

        mascara_obj = mascara[y:y+h, x:x+w]
        mascara_obj = resize_with_padding(mascara_obj, size=500)

        carac = extrair_caracteristicas(mascara_obj)

        if carac is None:
            continue

        X = pd.DataFrame([carac], columns=modelo.feature_names_in_)
        classe = modelo.predict(X)[0]

        conf = 1.0
        if hasattr(modelo, "predict_proba"):
            conf = max(modelo.predict_proba(X)[0])

        if conf < 0.60:
            continue

        if classe == "moeda":
            moedas += 1
            total_geral_moedas += 1
        else:
            lapis += 1
            total_geral_lapis += 1

        print("===================")
        print(f"Objeto em: ({x}, {y})")
        print(f"Bounding Box: {w}x{h}")
        print(f"Classe: {classe.capitalize()} | area= {int(area)} | conf = {conf:.2f}")
        print("---------------------------------")

        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            resultado,
            f"{classe.upper()} {conf:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    total = moedas + lapis

    cv2.rectangle(resultado, (10, 10), (240, 95), (255, 255, 255), -1)
    cv2.putText(resultado, f"Moedas: {moedas}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(resultado, f"Lapis: {lapis}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(resultado, f"Total: {total}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if moedas == 3 and lapis == 2:
        texto_status = "APROVADO"
        cor_status = (0, 255, 0) 
    else:
        texto_status = "REPROVADO"
        cor_status = (0, 0, 255) 

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.0
    espessura = 2

    (largura_texto, altura_texto), _ = cv2.getTextSize(texto_status, fonte, escala, espessura)

    x_status = resultado.shape[1] - largura_texto - 20
    y_status = 40

    cv2.rectangle(resultado, 
                  (x_status - 10, y_status - altura_texto - 10), 
                  (x_status + largura_texto + 10, y_status + 10), 
                  (255, 255, 255), -1)

    cv2.putText(resultado, texto_status, (x_status, y_status), fonte, escala, cor_status, espessura)

    cv2.imshow("Mascara", mascara)
    cv2.imshow("Resultado", resultado)

    while True:
        tecla = cv2.waitKey(0) & 0xFF

        if tecla in (ord('e'), ord('E')):
            break

        if tecla in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            print("\nPrograma encerrado pelo usuário.")
            exit()

    indice += 1

cv2.destroyAllWindows()

total_geral_objetos = total_geral_moedas + total_geral_lapis

print("\n=================================")
print("  RESUMO FINAL (TODAS AS IMAGENS)  ")
print("=================================")
print(f"Total de Moedas encontradas: {total_geral_moedas}")
print(f"Total de Lápis encontrados:  {total_geral_lapis}")
print("---------------------------------")
print(f"TOTAL GERAL DE OBJETOS:      {total_geral_objetos}")
print("=================================\n")