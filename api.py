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
CORS(app)

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
modelo_path = "/modelo_mixto2.keras"

try:
    modelo = tf.keras.models.load_model("modelo_mixto2.keras")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

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
        print(f"Error al centrar imagen: {e}")
        return np.zeros(size)

def base64_to_image(base64_str):
    """Convertir imagen base64 a numpy array"""
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert('L')
        return np.array(image)
    except Exception as e:
        print(f"Error al convertir imagen base64: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensaje": "API de Predicción de Dígitos",
        "status": "funcionando"
    })

@app.route('/guardar', methods=['POST'])
def guardar():
    """Guardar datos en base de datos"""
    if not conexion:
        return jsonify({"error": "No hay conexión a la base de datos"}), 500
    
    try:
        data = request.get_json()
        valor = data.get('valor')
        factorial = data.get('factorial')
        nombre_estudiante = data.get('nombre_estudiante')
        
        if not all([valor, factorial, nombre_estudiante]):
            return jsonify({"error": "Datos incompletos"}), 400
        
        cursor = conexion.cursor()
        sql = "INSERT INTO segundo_parcial (valor, factorial, nombre_estudiante) VALUES (%s, %s, %s)"
        cursor.execute(sql, (valor, factorial, nombre_estudiante))
        conexion.commit()
        cursor.close()
        
        return jsonify({"mensaje": "Datos guardados exitosamente"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predecirmas', methods=['POST'])
def predecir_multiple():
    """Predecir múltiples dígitos en una imagen"""
    if not modelo:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
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

@app.route('/modelo', methods=['GET'])
def verificar_modelo():
    """Verificar estado del modelo"""
    try:
        temp_modelo = tf.keras.models.load_model("modelo_mixto2.keras")
        return jsonify({
            "mensaje": "Modelo cargado correctamente", 
            "detalles": {
                "input_shape": temp_modelo.input_shape,
                "output_shape": temp_modelo.output_shape
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predecir', methods=['POST'])
def predecir():
    """Predecir dígito en una imagen"""
    if not modelo:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
        data = request.get_json()
        if not data or 'imagen' not in data:
            return jsonify({'error': 'Falta el campo imagen'}), 400

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

        return jsonify({
            'prediccion': prediccion,
            'confianza': round(confianza, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
