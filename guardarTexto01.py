import keras_ocr
import os
import csv
import numpy as np
from PIL import Image
import gc

carpeta_base = './imagenes'
MAX_SIZE = 800 
BATCH_SIZE = 1  

def redimensionar_imagen(image, max_size=MAX_SIZE):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        ratio = max_size / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil_img)
    return image

def procesar_imagen(path, pipeline):
    try:
        img = keras_ocr.tools.read(path)
        img = redimensionar_imagen(img)
        predictions = pipeline.recognize([img])[0]
        texto = ' '.join([word for word, box in predictions])
        del img
        gc.collect()
        return texto
    except Exception as e:
        print(f"Error procesando {path}: {str(e)}")
        return None

def procesar_carpeta(carpeta_imagenes, nombre_salida):
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    imagenes_ruta = [os.path.join(carpeta_imagenes, f) 
                    for f in os.listdir(carpeta_imagenes) 
                    if f.lower().endswith(extensiones_validas)]
    if not imagenes_ruta:
        print(f"\nNo se encontraron imágenes válidas en {carpeta_imagenes}")
        return
    try:
        pipeline = keras_ocr.pipeline.Pipeline()
    except Exception as e:
        print(f"Error al inicializar OCR: {str(e)}")
        return

    with open(f'resultados_{nombre_salida}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Nombre de archivo', 'Texto detectado'])
        for path in imagenes_ruta:
            print(f"\nProcesando: {os.path.basename(path)}...")
            texto = procesar_imagen(path, pipeline)
            if texto:
                writer.writerow([os.path.basename(path), texto])
                print(f"Texto detectado:\n{texto}")
            else:
                writer.writerow([os.path.basename(path), "ERROR EN PROCESAMIENTO"])

if __name__ == "__main__":
    carpetas = {
        'train': os.path.join(carpeta_base, 'train'),
        'validation': os.path.join(carpeta_base, 'validation')
    }
    for nombre, carpeta in carpetas.items():
        if os.path.exists(carpeta):
            print(f"\n{'='*50}\nProcesando {nombre.upper()}...\n{'='*50}")
            procesar_carpeta(carpeta, nombre)
        else:
            print(f"\nCarpeta no encontrada: {carpeta}")