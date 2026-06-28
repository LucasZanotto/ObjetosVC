# 🪙✏️ Sistema de Visão Computacional para Reconhecimento de Objetos

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

</p>

---

## 📖 Sobre o projeto

Este projeto foi desenvolvido para a disciplina de **Visão Computacional** e consiste em um sistema capaz de reconhecer automaticamente **moedas** e **lápis** utilizando técnicas de **Processamento Digital de Imagens** e **Machine Learning**.

O funcionamento é inspirado em sistemas industriais de **Inspeção Óptica Automática (AOI)**.

Ao analisar uma imagem o sistema:

✅ Detecta objetos

✅ Segmenta cada objeto

✅ Extrai características

✅ Classifica utilizando Inteligência Artificial

✅ Conta moedas e lápis

✅ Aprova ou reprova automaticamente a inspeção

---

# 🎯 Objetivo

Desenvolver um sistema completo de Visão Computacional contendo:

- 🖼 Processamento de Imagens
- 🤖 Inteligência Artificial
- 📊 Análise de desempenho
- 🖥 Interface gráfica
- 🔍 Reconhecimento de objetos

---

# 📂 Estrutura do Projeto

```text
ObjetosVC
│
├── dataset
│   ├── treino
│   │   ├── moeda
│   │   └── lapis
│   │
│   ├── teste
│   │   ├── moeda
│   │   └── lapis
│   │
│   └── multiplos
│
├── modelos
│   ├── caracteristicas.csv
│   └── modelo.pkl
│
├── src
│   ├── funcoes.py
│   ├── treinamento.py
│   ├── modelo.py
│   ├── detectar.py
│   └── detectar_multiplos.py
│
└── README.md
```

---

# 🧠 Pipeline do Sistema

```text
Imagem

   │

   ▼

Pré-processamento

   │

   ▼

Segmentação

   │

   ▼

Extração de Características

   │

   ▼

Random Forest

   │

   ▼

Classificação

   │

   ▼

Inspeção Final
```

---

# ⚙️ Técnicas utilizadas

## 🖼 Processamento de Imagem

- ✔ Conversão para escala de cinza
- ✔ CLAHE
- ✔ Gaussian Blur
- ✔ Threshold
- ✔ Otsu
- ✔ Canny
- ✔ Dilatação
- ✔ Abertura Morfológica
- ✔ Fechamento Morfológico
- ✔ Detecção de Contornos

---

## 📏 Características extraídas

O sistema não utiliza os pixels diretamente.

Cada objeto é representado por um vetor contendo:

| Característica |
|----------------|
| Área |
| Perímetro |
| Aspect Ratio |
| Extent |
| Solidity |
| Circularidade |
| Hu Moments |
| Convex Hull |
| Número de vértices |
| Convexity Defects |

Essas características alimentam o algoritmo de Machine Learning.

---

# 🤖 Inteligência Artificial

O algoritmo utilizado foi:

```text
Random Forest Classifier
```

Biblioteca:

```
Scikit-Learn
```

Treinamento:

```
81 imagens
```

- 🪙 40 moedas
- ✏️ 41 lápis

---

# 📊 Dataset

## Treinamento

| Classe | Quantidade |
|---------|-----------:|
| 🪙 Moedas | 40 |
| ✏️ Lápis | 41 |

Total:

```
81 imagens
```

---

## Teste

| Classe | Quantidade |
|---------|-----------:|
| 🪙 Moedas | 10 |
| ✏️ Lápis | 10 |

---

## Teste Final

Cada imagem contém vários objetos simultaneamente.

O sistema precisa:

- localizar todos os objetos;
- classificá-los;
- contabilizar as quantidades;
- validar a inspeção.

---

# 🖥 Interface

Durante a execução são exibidas duas janelas.

🟢 Máscara Segmentada

Mostra somente os objetos encontrados.

🟢 Resultado

Exibe:

- Bounding Boxes
- Classe
- Confiança
- Quantidade
- Aprovação

---

# ✔ Critério de Aprovação

Para ser considerada **APROVADA**, uma imagem deve conter exatamente:

| Objeto | Quantidade |
|---------|-----------:|
| 🪙 Moedas | 3 |
| ✏️ Lápis | 2 |

Caso contrário:

```
❌ REPROVADO
```

---

# 🚀 Como executar

## 1️⃣ Instale as dependências

```bash
pip install opencv-python numpy pandas scikit-learn joblib
```

---

## 2️⃣ Extraia as características

```bash
python treinamento.py
```

Será criado:

```
modelos/caracteristicas.csv
```

---

## 3️⃣ Treine o modelo

```bash
python modelo.py
```

Será criado:

```
modelo.pkl
```

---

## 4️⃣ Teste o modelo

```bash
python detectar.py
```

Será exibida a acurácia do classificador.

---

## 5️⃣ Execute a inspeção

```bash
python detectar_multiplos.py
```

---

# 📈 Exemplo de saída

```text
Contornos encontrados: 5

Objeto em: (429,753)

Classe: Moeda

Confiança: 1.00

----------------------------

Objeto em: (31,349)

Classe: Lápis

Confiança: 0.65
```

Resumo:

```text
=================================

Total de Moedas: 15

Total de Lápis: 9

Total Geral: 24

=================================
```

---

# 💡 Melhorias futuras

- [ ] Interface em PyQt
- [ ] Suporte para novos objetos
- [ ] Treinamento automático
- [ ] Uso de Redes Neurais (CNN)
- [ ] Exportação de relatórios
- [ ] Captura em tempo real pela webcam

---

# 👨‍💻 Autor

**Lucas Zanotto**

Projeto desenvolvido para a disciplina de **Visão Computacional**.

⭐ Se este projeto foi útil para você, deixe uma estrela no repositório!
