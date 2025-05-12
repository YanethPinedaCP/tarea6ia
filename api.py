from flask import Flask, request, jsonify, Response
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
import logging
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configuración de CORS más permisiva
CORS(app, resources={r"/*": {"origins": "*"}})

# Endpoint raíz con información del servicio
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "service": "IA Digit Recognition API",
        "endpoints": [
            "/predecir - Predict single digit",
            "/predecirmas - Predict multiple digits",
            "/modelo - Check model status"
        ]
    }), 200

# Endpoint de prueba
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "API is running smoothly"
    }), 200

# Configuración de base de datos
try:
    conexion = pymysql.connect(
        host=os.getenv('DB_HOST', 'www.server.daossystem.pro'),
        user=os.getenv('DB_USER', 'usr_ia_lf_2025'),
        password=os.getenv('DB_PASSWORD', '5sr_31_lf_2025'),
        db=os.getenv('DB_NAME', 'bd_ia_lf_2025'),
        port=int(os.getenv('DB_PORT', 3301))
    )
    logger.info("Conexión a base de datos establecida exitosamente")
except Exception as e:
    logger.error(f"Error de conexión a la base de datos: {e}")
    conexion = None

# Cargar modelo con manejo de errores
try:
    modelo = tf.keras.models.load_model("modelo_mixto2.keras")
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    modelo = None

# Manejador de preflight para CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Funciones de procesamiento de imagen
def centrar_imagen(imagen_np, size=(28, 28)):
    """Centrar y normalizar imagen para predicción"""
    try:
        imagen_np = np.where(imagen_np < 0.2, 0, imagen_np)
        coords = np.column_stack(np.where(imagen_np > 0.1))
        
        if coords.size == 0:
            return np.zeros(size)
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        recorte = imagen_np[y0:y1+1, x0:x1+1]
        
        imagen_recortada = Image.fromarray((recorte * 255).astype(np.uint8))
        imagen_centrada = ImageOps.pad(imagen_recortada, size, color=0, centering=(0.5, 0.5))
        
        return np.array(imagen_centrada) / 255.0
    except Exception as e:
        logger.error(f"Error al centrar imagen: {e}")
        return np.zeros(size)

def base64_to_image(base64_str):
    """Convertir imagen base64 a numpy array"""
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert('L')
        return np.array(image)
    except Exception as e:
        logger.error(f"Error al convertir imagen base64: {e}")
        return None

# Endpoint de predicción múltiple
@app.route('/predecirmas', methods=['POST', 'OPTIONS'])
def predecir_multiple():
    # Manejo de preflight request
    if request.method == 'OPTIONS':
        return Response(status=200)

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
        logger.error(f"Error en /predecirmas: {e}")
        return jsonify({'error': str(e)}), 500

# Manejador de errores 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/health",
            "/predecir",
            "/predecirmas",
            "/modelo"
        ]
    }), 404

# Añade manejo de puerto para Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
