#!/usr/bin/env python3
"""
Segmentação de Embalagens de Produtos Avícolas
Trabalho Prático 1 - Processamento Digital de Imagens (PDI)
IFG - Instituto Federal de Goiás

Este script percorre todas as imagens nas pastas do conjunto de dados,
detecta as regiões correspondentes aos rótulos/embalagens que contêm
o nome do produto, e salva os recortes dessas regiões no diretório de saída.
"""

import os
import sys
import argparse
import cv2
import numpy as np

def segment_image(img_path, output_dir, img_name):
    # Carrega a imagem original em escala de cinza para detecção
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return 0

    # Se a imagem tiver canais de cor, carregamos também a colorida para salvar o crop colorido
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 1. Definição da Região de Interesse (ROI) horizontal
    mask = np.zeros_like(img)
    mask[:, 250:1030] = 255
    masked_img = cv2.bitwise_and(img, mask)

    pixels_inside = img[:, 250:1030].flatten()
    if len(pixels_inside) == 0:
        return 0

    candidates = []

    # 2. Limiarização Dinâmica com Fallback
    # Tentamos encontrar candidatos primeiro com Percentile 90, abertura 3x3 e fechamento 15x15.
    # Se falhar, caímos para percentis mais baixos para garantir detecção em condições extremas de luz.
    for p in [90, 85]:
        thresh_val = np.percentile(pixels_inside, p)
        _, thresh = cv2.threshold(masked_img, thresh_val, 255, cv2.THRESH_BINARY)

        # 3. Operação Morfológica (Abertura e depois Fechamento)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 4. Extração de Contornos
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            cx = x + w/2

            # 5. Filtragem Geométrica Básica
            if w >= 100 and h >= 100 and area >= 18000 and w < 380 and h < 380:
                if 250 < cx < 1030:
                    aspect = w / float(h)
                    # Filtra razões de aspecto inválidas (reflexos ou rugas muito estreitos/largos)
                    if 0.45 <= aspect <= 2.2:
                        # Exige que não esteja colado nas bordas da imagem ou do ROI
                        if not (x <= 265 or (x + w) >= 1015 or y <= 15 or (y + h) >= 705):
                            # Filtro de Componentes Conectados (mínimo 25 componentes de tamanho >= 4px)
                            crop_thresh = thresh[y:y+h, x:x+w]
                            num_labels, labels_im = cv2.connectedComponents(crop_thresh)
                            unique, counts = np.unique(labels_im, return_counts=True)
                            comp_ge_4 = 0
                            for val, count in zip(unique, counts):
                                if val == 0:
                                    continue
                                if count >= 4:
                                    comp_ge_4 += 1

                            if comp_ge_4 >= 25:
                                crop_gray = img[y:y+h, x:x+w]
                                min_val = crop_gray.min()
                                
                                # Rejeita marca do fabricante (Super Frango) e reflexos claros com min_val > 80
                                if min_val <= 80:
                                    std_val = crop_gray.std()
                                    mean_val = crop_gray.mean()
                                    lap_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()

                                    # Score com escala exponencial de min_val para favorecer rótulos com letras escuras e nítidas
                                    score = area * mean_val * lap_var * std_val * np.exp((255.0 - min_val) / 6.0)

                                    candidates.append({
                                        'box': (x, y, w, h),
                                        'score': score
                                    })

        if len(candidates) > 0:
            break

    if len(candidates) == 0:
        return 0

    # 7. Seleção do Melhor Candidato
    best_cand = max(candidates, key=lambda x: x['score'])
    x, y, w, h = best_cand['box']

    # 8. Margem de Segurança (Padding) de 20 pixels
    pad = 20
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + w + pad)

    crop_to_save = img_color[y1:y2, x1:x2] if img_color is not None else img[y1:y2, x1:x2]

    # Salva a imagem segmentada
    base_name, _ = os.path.splitext(img_name)
    out_name = f"{base_name}_segmentada_1.png"
    out_name = out_name.replace(":", "_")
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, crop_to_save)

    return 1

def process_dataset(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    print(f"Iniciando o processamento.")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída:   {output_dir}")

    total_images = 0
    total_crops = 0

    # Percorre todas as pastas de classes
    subdirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for subdir in subdirs:
        class_in_path = os.path.join(input_dir, subdir)
        class_out_path = os.path.join(output_dir, subdir)
        
        # Cria a pasta correspondente no resultado se não existir
        os.makedirs(class_out_path, exist_ok=True)
        
        # Filtra os arquivos de imagem na pasta
        files = sorted([f for f in os.listdir(class_in_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nProcessando classe: {subdir} ({len(files)} imagens)")
        
        class_crops = 0
        for f in files:
            img_path = os.path.join(class_in_path, f)
            crops = segment_image(img_path, class_out_path, f)
            class_crops += crops
            total_images += 1
            
        print(f"  -> Concluído! {class_crops} recortes gerados.")
        total_crops += class_crops

    print(f"\n==========================================")
    print(f"Processamento concluído com sucesso!")
    print(f"Total de imagens processadas: {total_images}")
    print(f"Total de recortes gerados:    {total_crops}")
    print(f"Os resultados estão salvos em: {output_dir}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentação de Embalagens de Produtos Avícolas (PDI)")
    parser.add_argument("--input", default="Train_and_Validation", help="Diretório com as imagens de entrada")
    parser.add_argument("--output", default="resultado", help="Diretório onde os resultados serão salvos")
    args = parser.parse_args()

    process_dataset(args.input, args.output)
