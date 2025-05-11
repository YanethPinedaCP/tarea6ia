from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation
from flask_cors import CORS
import matplotlib.pyplot as plt
import cv2
import pymysql


app = Flask(__name__)
CORS(app) 
conexion = pymysql.connect(
    host='www.server.daossystem.pro',
    user='usr_ia_lf_2025',
    password='5sr_31_lf_2025',
    db='bd_ia_lf_2025',
    port=3301
)
# Cargar el modelo entrenado una sola vez
modelo = tf.keras.models.load_model("modelo_mixto2.keras")
# =========================================
# Función para centrar imagen
# =========================================
def centrar_imagen(imagen_np, size=(28, 28)):
    imagen_np = np.where(imagen_np < 0.2, 0, imagen_np)  # Reducir ruido
    coords = np.column_stack(np.where(imagen_np > 0.1))  # Ajustar el umbral
    if coords.size == 0:
        return np.zeros(size)  # Imagen vacía
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    recorte = imagen_np[y0:y1+1, x0:x1+1]
    imagen_recortada = Image.fromarray((recorte * 255).astype(np.uint8))
    imagen_centrada = ImageOps.pad(imagen_recortada, size, color=0, centering=(0.5, 0.5))
    return np.array(imagen_centrada) / 255.0

def base64_to_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes)).convert('L')
    return np.array(image)

@app.route('/', methods=['GET'])
def home():
    return "Hola mundo estamos consumiendo esta api"


@app.route('/guardar', methods=['POST'])
def guardar():
    data = request.get_json()
    valor = data['valor']
    factorial = data['factorial']
    nombre_estudiante = data['nombre_estudiante']
    
    cursor = conexion.cursor()
    sql = "INSERT INTO segundo_parcial (valor, factorial, nombre_estudiante) VALUES (%s, %s, %s)"
    cursor.execute(sql, (valor, factorial, nombre_estudiante))
    conexion.commit()
    cursor.close()
    
    return jsonify({"mensaje": "Datos guardados exitosamente"})


@app.route('/predecirmas', methods=['POST'])
def predecir2():
    data = request.get_json()
    if 'imagen' not in data:
        return jsonify({'error': 'Falta el campo imagen'}), 400
    try:
        base64_str = data['imagen']
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        imagen_np = base64_to_image(base64_str)

        # Invertir si fondo blanco
        if np.mean(imagen_np) > 127:
            imagen_np = 255 - imagen_np

        # Binarizar y dilatar
        _, binaria = cv2.threshold(imagen_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria = cv2.dilate(binaria, np.ones((2, 2), np.uint8), iterations=1)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])

        print("Contornos detectados:", len(contornos))
        resultados = []

        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            if w * h < 50:
                continue

            recorte = imagen_np[y:y+h, x:x+w]
            recorte = recorte / 255.0
            recorte = centrar_imagen(recorte)
            entrada = recorte.reshape(1, 28, 28, 1)

            pred = modelo.predict(entrada)
            digito = np.argmax(pred)
            confianza = np.max(pred) * 100

            resultados.append({
                'digito': int(digito),
                'confianza': float(round(confianza, 2)),
                'coordenadas': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })

        # Mostrar número final
        print("Número reconocido:", ''.join(str(r['digito']) for r in resultados))

        return jsonify({'resultados': resultados})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# =========================================
# Endpoint de predicción
# =========================================
@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    if not data or 'imagen' not in data:
        return jsonify({'error': 'Falta el campo imagen'}), 400

    try:
        # Convertir base64 a imagen PIL
        base64_str = data['imagen']
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        imagen_bytes = base64.b64decode(base64_str)
        
        imagen = Image.open(BytesIO(imagen_bytes)).convert('L')

        imagen = ImageOps.invert(imagen)
        # Procesar imagen
        imagen_np = np.array(imagen) / 255.0
        imagen_np = binary_dilation(imagen_np, iterations=1).astype(float)
        imagen_np = centrar_imagen(imagen_np)
        

        # Preparar para el modelo
        imagen_input = imagen_np.reshape(1, 28, 28, 1)


        # Hacer predicción
        predicciones = modelo.predict(imagen_input)
        prediccion = int(np.argmax(predicciones))
        confianza = float(np.max(predicciones) * 100)
        print("Confianza", {confianza})
        print("Prediccion", {prediccion})
        return jsonify({
            'prediccion': prediccion,
            'confianza': round(confianza, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================================
# Ejecutar el servidor
# =========================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
