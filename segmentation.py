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
    # As imagens têm resolução 1280x720. As bordas laterais (x < 250 e x > 1030)
    # contêm a esteira transportadora e as abas da caixa de papelão (onde há logotipos de marcas).
    # Criamos uma máscara para analisar apenas o centro onde ficam as embalagens, evitando falsos positivos.
    mask = np.zeros_like(img)
    mask[:, 250:1030] = 255
    masked_img = cv2.bitwise_and(img, mask)

    # 2. Binarização Dinâmica por Percentil
    # Como as condições de iluminação variam e os rótulos são sempre as partes
    # mais claras da imagem dentro da caixa, calculamos o percentil 90 dos pixels da ROI.
    # Isso define um limiar adaptativo de alta intensidade para cada imagem individualmente.
    pixels_inside = img[:, 250:1030].flatten()
    if len(pixels_inside) == 0:
        return 0
    thresh_val = np.percentile(pixels_inside, 90)
    _, thresh = cv2.threshold(masked_img, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Operação Morfológica (Fechamento)
    # Rótulos contêm textos e códigos de barras (pixels escuros sobre fundo claro).
    # O fechamento morfológico com um elemento estruturante retangular (15x15)
    # preenche os "buracos" pretos do texto e unifica as letras em blocos sólidos de máscara.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Extração de Contornos
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crop_count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        cx = x + w/2

        # 5. Filtragem por Tamanho, Área e Posição (na caixa central)
        # Rótulos reais de produtos avícolas têm tamanhos previsíveis na imagem:
        # - Largura e altura entre 60 e 350 pixels.
        # - Área total entre 4.000 e 80.000 pixels quadrados.
        # - O centro do rótulo deve estar dentro do limite central (250 < cx < 1030).
        if w >= 60 and h >= 60 and area >= 4000 and w < 350 and h < 350:
            if 250 < cx < 1030:
                # Extrai a região candidata não acolchoada para verificação de textura/contraste
                crop_gray_unpadded = img[y:y+h, x:x+w]
                
                # 6. Filtragem por Textura, Contraste e Relação de Pixels Escuros (Falsos Positivos)
                # Para evitar falsos positivos vindos de reflexos plásticos ou brilhos lisos na caixa:
                # A) Canny edge ratio >= 0.04 (deve conter detalhes texturizados)
                # B) Desvio padrão >= 15.0 (deve apresentar bom contraste local)
                # C) Proporção de pixels escuros (< 120) >= 0.05 (deve conter texto impresso)
                std_val = crop_gray_unpadded.std()
                dark_ratio = np.sum(crop_gray_unpadded < 120) / float(area)
                
                crop_edges = cv2.Canny(crop_gray_unpadded, 30, 90)
                edge_ratio = np.sum(crop_edges > 0) / float(area)

                if edge_ratio >= 0.04 and std_val >= 15.0 and dark_ratio >= 0.05:
                    crop_count += 1
                    
                    # 7. Margem de Segurança (Padding) de 20 pixels
                    # Adicionamos uma margem de segurança ao redor do contorno para garantir
                    # que nenhuma palavra ou texto nas bordas do rótulo/cinta seja cortado.
                    pad = 20
                    y1 = max(0, y - pad)
                    y2 = min(img.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(img.shape[1], x + w + pad)
                    
                    # Realiza o crop da imagem colorida se disponível, senão da cinza
                    crop_to_save = img_color[y1:y2, x1:x2] if img_color is not None else img[y1:y2, x1:x2]
                    
                    # Salva a imagem segmentada
                    base_name, _ = os.path.splitext(img_name)
                    out_name = f"{base_name}_segmentada_{crop_count}.png"
                    out_name = out_name.replace(":", "_")
                    out_path = os.path.join(output_dir, out_name)
                    cv2.imwrite(out_path, crop_to_save)

    return crop_count

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
