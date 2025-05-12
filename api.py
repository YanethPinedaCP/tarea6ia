from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation
from flask_cors import CORS
import cv2
import pymysql
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configuración de CORS más permisiva
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://yanethpinedacp.github.io",  # Tu dominio específico
            "http://localhost:5000",  # Para desarrollo local
            "*"  # Usar con precaución, solo para pruebas
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuración de base de datos desde variables de entorno
try:
    conexion = pymysql.connect(
        host=os.getenv('DB_HOST', 'www.server.daossystem.pro'),
        user=os.getenv('DB_USER', 'usr_ia_lf_2025'),
        password=os.getenv('DB_PASSWORD', '5sr_31_lf_2025'),
        db=os.getenv('DB_NAME', 'bd_ia_lf_2025'),
        port=int(os.getenv('DB_PORT', 3301))
    )
except Exception as e:
    print(f"Error de conexión a la base de datos: {e}")
    conexion = None

# Cargar modelo con manejo de errores
try:
    modelo = tf.keras.models.load_model("modelo_mixto2.keras")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

# Manejador de preflight para CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://yanethpinedacp.github.io')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Resto del código anterior... (mantén todas las funciones anteriores)

@app.route('/predecirmas', methods=['POST', 'OPTIONS'])
def predecir_multiple():
    # Manejo de preflight request
    if request.method == 'OPTIONS':
        response = jsonify(success=True)
        response.headers.add('Access-Control-Allow-Origin', 'https://yanethpinedacp.github.io')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if not modelo:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
        # Asegúrate de que los datos sean JSON
        if not request.is_json:
            return jsonify({'error': 'Contenido debe ser JSON'}), 400

        data = request.get_json()
        if 'imagen' not in data:
            return jsonify({'error': 'Falta el campo imagen'}), 400

        base64_str = data['imagen']
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        imagen_np = base64_to_image(base64_str)
        if imagen_np is None:
            return jsonify({'error': 'Error al procesar imagen'}), 400

        if np.mean(imagen_np) > 127:
            imagen_np = 255 - imagen_np

        _, binaria = cv2.threshold(imagen_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaria = cv2.dilate(binaria, np.ones((2, 2), np.uint8), iterations=1)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])

        resultados = []

        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            if w * h < 50:
                continue

            recorte = imagen_np[y:y+h, x:x+w] / 255.0
            recorte = centrar_imagen(recorte)
            entrada = recorte.reshape(1, 28, 28, 1)

            pred = modelo.predict(entrada)
            digito = np.argmax(pred)
            confianza = np.max(pred) * 100

            resultados.append({
                'digito': int(digito),
                'confianza': round(confianza, 2),
                'coordenadas': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })

        return jsonify({'resultados': resultados})

    except Exception as e:
        print(f"Error en /predecirmas: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
