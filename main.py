from ultralytics import YOLO
import cv2
import numpy as np
import os
import re
import easyocr
from util import write_csv

# Inicializar EasyOCR con configuración optimizada para placas ecuatorianas
print("Inicializando EasyOCR para placas ecuatorianas...")
reader = easyocr.Reader(
    ['en'], 
    gpu=False,
    model_storage_directory='./ocr_model',
    download_enabled=True,
    recognizer='en',
    detector='dbnet18'
)
print("✅ EasyOCR listo")

# Cargar modelos
print("Cargando modelos YOLO...")
coco_model = YOLO("yolov8n.pt")
licence_plate_detector = YOLO('./models/yolov8n_combinated_best.pt')
print("✅ Modelos cargados correctamente")

def preprocess_for_ocr(image):
    """Preprocesamiento específico para placas ecuatorianas"""
    results = []
    
    # Convertir a gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 1. Original
    results.append(('original', gray))
    
    # 2. Escalar (las placas pequeñas mejoran)
    if gray.shape[0] < 50 or gray.shape[1] < 150:
        scale_factor = max(2, 300 / gray.shape[1])
        scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        results.append(('scaled', scaled))
    
    # 3. CLAHE (mejora contraste local)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    clahe_img = clahe.apply(gray)
    results.append(('clahe', clahe_img))
    
    # 4. Binarización adaptativa
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 15, 8)
    results.append(('adaptive', binary_adaptive))
    
    # 5. Otsu thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(('otsu', otsu))
    
    # 6. Morfología para limpiar ruido
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    results.append(('morph', morphed))
    
    # 7. Sharpening
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    results.append(('sharp', sharpened))
    
    # 8. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    results.append(('denoise', denoised))
    
    # 9. Gamma correction
    gamma = 1.3
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    results.append(('gamma', gamma_corrected))
    
    return results

def clean_plate_text_ecuador(text):
    """Limpiar y normalizar texto de placa ecuatoriana"""
    if not text:
        return None
    
    # Limpiar caracteres no alfanuméricos
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Correcciones comunes de OCR para placas ecuatorianas
    # Formato Ecuador: ABC-1234 (3 letras, 4 números) o ABC1234
    corrections = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G',
        'I': '1', 'O': '0', 'Z': '2', 'S': '5', 'B': '8',
        'Q': 'O', 'U': 'V', 'D': '0', 'P': 'R', 'C': 'G'
    }
    
    # Aplicar correcciones según el formato ecuatoriano
    if len(text) >= 7:
        text_list = list(text)
        # Las primeras 3 posiciones son letras
        for i in range(min(3, len(text_list))):
            if text_list[i].isdigit():
                text_list[i] = corrections.get(text_list[i], text_list[i])
        # Las últimas 4 posiciones son números
        for i in range(3, min(7, len(text_list))):
            if text_list[i].isalpha():
                text_list[i] = corrections.get(text_list[i], text_list[i])
        text = ''.join(text_list)
    
    # Validar formato ecuatoriano (7-8 caracteres)
    if len(text) == 7:
        # Formato: 3 letras + 4 números
        if re.match(r'^[A-Z]{3}\d{4}$', text):
            return text
        # Intentar corregir formato común
        elif re.match(r'^[A-Z0-9]{7}$', text):
            # Asegurar 3 letras al inicio y 4 números al final
            first_three = ''.join([c if c.isalpha() else corrections.get(c, 'A') for c in text[:3]])
            last_four = ''.join([c if c.isdigit() else corrections.get(c, '0') for c in text[3:7]])
            text = first_three + last_four
            if re.match(r'^[A-Z]{3}\d{4}$', text):
                return text
    
    elif len(text) == 8:
        # Formato: 3 letras + 4 números + 1 letra (algunos casos)
        if re.match(r'^[A-Z]{3}\d{4}[A-Z]$', text):
            return text
    
    # Si no cumple formato, retornar None
    return None

def read_license_plate_enhanced(plate_crop):
    """Versión mejorada de lectura de placas para Ecuador"""
    best_text = None
    best_score = 0
    best_method = None
    
    # Probar diferentes preprocesamientos
    for method_name, processed_img in preprocess_for_ocr(plate_crop):
        try:
            # Configuración específica para EasyOCR
            result = reader.readtext(
                processed_img,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                width_ths=0.5,
                height_ths=0.5,
                decoder='greedy',
                beamWidth=5
            )
            
            for detection in result:
                bbox, text, score = detection
                text_clean = clean_plate_text_ecuador(text)
                
                if text_clean:
                    # Bonus por formato correcto de Ecuador
                    if re.match(r'^[A-Z]{3}\d{4}$', text_clean):
                        score += 0.4
                    elif re.match(r'^[A-Z]{3}\d{4}[A-Z]$', text_clean):
                        score += 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_text = text_clean
                        best_method = method_name
                        
        except Exception as e:
            continue
    
    return best_text, best_score

# Configuración
vehicles = [2, 3, 5, 7]
results = {}
input_folder = "./imagenes_entrada"
output_folder = "./resultados"

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'recortes'), exist_ok=True)

imagenes = [f for f in os.listdir(input_folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

print(f"\n📸 Procesando {len(imagenes)} imagen(es)...\n")

# Estadísticas
total_detecciones = 0
imagenes_con_placas = 0
ocr_fallos = 0

for idx, imagen_nombre in enumerate(imagenes, 1):
    print(f"[{idx}/{len(imagenes)}] Procesando: {imagen_nombre}")
    
    frame = cv2.imread(os.path.join(input_folder, imagen_nombre))
    if frame is None:
        continue
    
    altura_original, ancho_original = frame.shape[:2]
    
    # Detectar placas
    detecciones_totales = []
    
    # Escala original con menor confianza
    results_orig = licence_plate_detector(frame, conf=0.15, iou=0.3)[0]
    for det in results_orig.boxes.data.tolist():
        detecciones_totales.append(det)
    
    # Escala más grande
    if max(altura_original, ancho_original) < 1920:
        frame_big = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        results_big = licence_plate_detector(frame_big, conf=0.15, iou=0.3)[0]
        for det in results_big.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            x1, y1, x2, y2 = x1/1.5, y1/1.5, x2/1.5, y2/1.5
            detecciones_totales.append([x1, y1, x2, y2, score, class_id])
    
    # Mejorar contraste de la imagen completa
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    results_enhanced = licence_plate_detector(frame_enhanced, conf=0.15, iou=0.3)[0]
    for det in results_enhanced.boxes.data.tolist():
        detecciones_totales.append(det)
    
    # NMS para eliminar duplicados
    if detecciones_totales:
        detecciones_totales = sorted(detecciones_totales, key=lambda x: x[4], reverse=True)
        detecciones_finales = []
        
        for det in detecciones_totales:
            x1, y1, x2, y2, score, class_id = det
            mantener = True
            for det_final in detecciones_finales:
                fx1, fy1, fx2, fy2, fscore, fclass = det_final
                ix1 = max(x1, fx1)
                iy1 = max(y1, fy1)
                ix2 = min(x2, fx2)
                iy2 = min(y2, fy2)
                if ix2 > ix1 and iy2 > iy1:
                    iou = (ix2-ix1)*(iy2-iy1) / ((x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - (ix2-ix1)*(iy2-iy1))
                    if iou > 0.4:
                        mantener = False
                        break
            if mantener:
                detecciones_finales.append(det)
    else:
        detecciones_finales = []
    
    # Procesar detecciones
    resultados_imagen = {}
    placa_contador = 0
    
    for placa in detecciones_finales:
        if len(placa) >= 6:
            x1, y1, x2, y2, score, class_id = placa[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Validar coordenadas
            if x1 < 0 or y1 < 0 or x2 > ancho_original or y2 > altura_original:
                continue
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 30 or (y2-y1) < 10:
                continue
            
            # Recortar placa con margen
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(ancho_original, x2 + margin)
            y2 = min(altura_original, y2 + margin)
            
            placa_crop = frame[y1:y2, x1:x2]
            if placa_crop.size == 0:
                continue
            
            # Guardar recorte para depuración
            recorte_path = os.path.join(output_folder, 'recortes', f"recorte_{imagen_nombre}_{placa_contador+1}.jpg")
            cv2.imwrite(recorte_path, placa_crop)
            
            # Leer placa
            texto_placa, texto_score = read_license_plate_enhanced(placa_crop)
            
            if texto_placa:
                placa_contador += 1
                resultados_imagen[placa_contador] = {
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': texto_placa,
                        'bbox_score': score,
                        'text_score': texto_score if texto_score else 0.0
                    }
                }
                
                print(f"  🚗 Placa Ecuador: {texto_placa} (OCR: {texto_score:.2f}, Detección: {score:.2f})")
                
                # Dibujar resultados
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, texto_placa, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                ocr_fallos += 1
    
    # Guardar imagen
    if resultados_imagen:
        imagenes_con_placas += 1
        total_detecciones += placa_contador
        output_path = os.path.join(output_folder, f"resultado_{imagen_nombre}")
        cv2.imwrite(output_path, frame)
        print(f"  💾 Guardada: {placa_contador} placa(s)")
    else:
        print(f"  ⚠️ Sin placas detectadas")
    
    results[imagen_nombre] = resultados_imagen
    print()

# Guardar CSV
csv_path = os.path.join(output_folder, 'resultados.csv')
write_csv(results, csv_path)

print(f"\n{'='*50}")
print(f"✅ PROCESAMIENTO COMPLETADO")
print(f"{'='*50}")
print(f"📊 Estadísticas finales:")
print(f"   - Total imágenes: {len(imagenes)}")
print(f"   - Imágenes con placas: {imagenes_con_placas}")
print(f"   - Total placas detectadas: {total_detecciones}")
print(f"   - Tasa de detección: {(imagenes_con_placas/len(imagenes))*100:.1f}%")
print(f"   - Fallos de OCR: {ocr_fallos}")
print(f"\n📁 Resultados guardados en: {output_folder}")
print(f"📁 Recortes de placas guardados en: {output_folder}/recortes")

# Exportar resultados a TXT legible
with open(os.path.join(output_folder, 'placas_detectadas.txt'), 'w', encoding='utf-8') as f:
    f.write("RESULTADOS DE DETECCIÓN DE PLACAS ECUATORIANAS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total imágenes procesadas: {len(imagenes)}\n")
    f.write(f"Imágenes con placas: {imagenes_con_placas}\n")
    f.write(f"Total placas detectadas: {total_detecciones}\n")
    f.write(f"Tasa de éxito: {(imagenes_con_placas/len(imagenes))*100:.1f}%\n\n")
    f.write("="*60 + "\n\n")
    
    for img_name, img_results in results.items():
        if img_results:
            f.write(f"📷 {img_name}:\n")
            for plate_id, plate_data in img_results.items():
                text = plate_data['license_plate']['text']
                score = plate_data['license_plate']['text_score']
                f.write(f"   - Placa {plate_id}: {text} (confianza: {score:.2f})\n")
            f.write("\n")