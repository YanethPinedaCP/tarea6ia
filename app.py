from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation
from flask_cors import CORS
import cv2
import os
import logging
import sys

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Salida a consola
        logging.FileHandler('app.log')      # Registro en archivo
    ]
)

# Deshabilitar warnings de TensorFlow que no son críticos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
CORS(app)

# Función de carga de modelo con diagnóstico detallado
def load_model_with_diagnostics():
    try:
        logging.info("Iniciando carga del modelo...")
        
        # Imprimir directorio actual y contenido
        logging.info(f"Directorio actual: {os.getcwd()}")
        logging.info(f"Contenido del directorio: {os.listdir('.')}")
        
        # Verificar existencia del modelo
        modelo_path = "modelo_mixto2.keras"
        if not os.path.exists(modelo_path):
            logging.error(f"Archivo de modelo no encontrado: {modelo_path}")
            return None
        
        # Información del archivo de modelo
        model_stats = os.stat(modelo_path)
        logging.info(f"Tamaño del modelo: {model_stats.st_size} bytes")
        
        # Cargar modelo con verificaciones
        tf.keras.backend.set_learning_phase(0)
        modelo = tf.keras.models.load_model(modelo_path, compile=False)
        
        logging.info("Modelo cargado exitosamente")
        logging.info(f"Estructura del modelo: {modelo.summary()}")
        
        return modelo
    
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        logging.error(f"Tipo de error: {type(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# Cargar modelo al inicio
modelo = load_model_with_diagnostics()

# Resto de funciones de procesamiento
def centrar_imagen(imagen_np, size=(28, 28)):
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
        logging.error(f"Error en centrar_imagen: {e}")
        return np.zeros(size)

def base64_to_image(base64_str):
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert('L')
        return np.array(image)
    except Exception as e:
        logging.error(f"Error en base64_to_image: {e}")
        raise

# Endpoints
@app.route('/', methods=['GET'])
def home():
    return "API de Reconocimiento de Dígitos Funcionando"

@app.route('/predecir', methods=['POST'])
def predecir():
    logging.info("Endpoint de predicción invocado")
    
    if modelo is None:
        logging.error("Modelo no cargado")
        return jsonify({'error': 'Modelo no cargado'}), 500

    data = request.get_json()
    if not data or 'imagen' not in data:
        logging.warning("Falta campo de imagen")
        return jsonify({'error': 'Falta el campo imagen'}), 400

    try:
        base64_str = data['imagen']
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        imagen_bytes = base64.b64decode(base64_str)
        imagen = Image.open(BytesIO(imagen_bytes)).convert('L')
        imagen = ImageOps.invert(imagen)
        
        imagen_np = np.array(imagen) / 255.0
        imagen_np = binary_dilation(imagen_np, iterations=1).astype(float)
        imagen_np = centrar_imagen(imagen_np)
        
        imagen_input = imagen_np.reshape(1, 28, 28, 1)
        
        predicciones = modelo.predict(imagen_input)
        prediccion = int(np.argmax(predicciones))
        confianza = float(np.max(predicciones) * 100)
        
        logging.info(f"Predicción: {prediccion}, Confianza: {confianza}")
        
        return jsonify({
            'prediccion': prediccion,
            'confianza': round(confianza, 2)
        })

    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Configuración para despliegue
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logging.info(f"Iniciando servidor en puerto {port}")
    app.run(host='0.0.0.0', port=port)
