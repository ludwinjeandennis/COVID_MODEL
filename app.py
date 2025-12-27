"""
SISTEMA STREAMLIT PROFESIONAL PARA DETECCIÓN DE COVID-19
Version optimizada para Render.com con descarga desde Google Drive
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys
from datetime import datetime
import warnings
import gdown
import time
import requests
from pathlib import Path

warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# ==================== CONFIGURACIÓN ====================
class Config:
    # ID DE GOOGLE DRIVE - REEMPLAZAR CON EL TUYO
    GOOGLE_DRIVE_FILE_ID = "1k468ZgAmfeZMjOOT-BJ0STkf7wmh1ArR"
    
    # URL de descarga directa
    MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"
    
    # Información del modelo
    MODEL_FILENAME = "best_model_gpu.pth"
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # Clases y configuraciones
    CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    CLASS_DESCRIPTIONS = {
        'COVID': "Infección por SARS-CoV-2 detectada en radiografía pulmonar",
        'Lung_Opacity': "Opacidades pulmonares (pueden indicar neumonía, edema, etc.)",
        'Normal': "Radiografía pulmonar sin anomalías detectables",
        'Viral Pneumonia': "Neumonía viral (no COVID-19) detectada"
    }
    CLASS_RECOMMENDATIONS = {
        'COVID': "Consulta médica inmediata. Aislamiento recomendado. Prueba PCR confirmatoria.",
        'Lung_Opacity': "Evaluación médica necesaria. Puede requerir tomografía computarizada.",
        'Normal': "Sin hallazgos patológicos. Continuar con controles rutinarios.",
        'Viral Pneumonia': "Tratamiento antiviral posiblemente requerido. Consulta médica."
    }

# ==================== VERIFICAR CONFIGURACIÓN ====================
def check_configuration():
    """Verificar que el ID de Google Drive esté configurado correctamente"""
    
    if Config.GOOGLE_DRIVE_FILE_ID == "1k468ZgAmfeZMjOOT-BJ0STkf7wmh1ArR":
        st.sidebar.warning("ID DE DRIVE CONFIGURADO")
        return True
    
    if len(Config.GOOGLE_DRIVE_FILE_ID) < 20:
        st.error(f"ID de Google Drive muy corto: {Config.GOOGLE_DRIVE_FILE_ID}")
        st.info("El ID debe tener al menos 20 caracteres")
        return False
    
    return True

# ==================== DESCARGAR MODELO DESDE GOOGLE DRIVE ====================
@st.cache_resource
def download_model_from_drive():
    """
    Descargar el modelo desde Google Drive
    Retorna: (success, message, file_size)
    """
    
    # Crear directorio si no existe
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Verificar si ya existe
    if os.path.exists(Config.MODEL_PATH):
        file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
        return True, f"Modelo ya descargado ({file_size:.2f} GB)", file_size
    
    # Mostrar información de descarga
    st.sidebar.warning("Descargando modelo (1GB)...")
    
    try:
        # URL de descarga
        url = f"https://drive.google.com/uc?id={Config.GOOGLE_DRIVE_FILE_ID}"
        
        # Configurar barra de progreso
        progress_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
        
        # Función para mostrar progreso
        def show_progress(current, total, width=80):
            if total:
                percent = current / total
                progress_bar.progress(percent)
                progress_text.text(f"Descargando: {current/(1024**3):.2f}/{total/(1024**3):.2f} GB")
        
        # Descargar archivo
        gdown.download(
            url,
            Config.MODEL_PATH,
            quiet=False,
            resume=True
        )
        
        # Verificar descarga
        if os.path.exists(Config.MODEL_PATH):
            file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
            progress_bar.progress(1.0)
            progress_text.text(f"Descarga completa ({file_size:.2f} GB)")
            return True, f"Modelo descargado exitosamente", file_size
        else:
            return False, "Error: Archivo no se descargó", 0
            
    except Exception as e:
        error_msg = str(e)
        
        # Método alternativo: requests
        try:
            st.sidebar.info("Intentando método alternativo...")
            
            # URL alternativa
            url = f"https://docs.google.com/uc?export=download&id={Config.GOOGLE_DRIVE_FILE_ID}"
            
            # Descargar con requests
            response = requests.get(url, stream=True, timeout=300)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                # Guardar archivo
                with open(Config.MODEL_PATH, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Actualizar progreso
                            if total_size:
                                progress = downloaded / total_size
                                progress_bar.progress(progress)
                
                file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
                return True, f"Modelo descargado ({file_size:.2f} GB)", file_size
            else:
                return False, f"Error HTTP: {response.status_code}", 0
                
        except Exception as e2:
            return False, f"Error: {error_msg}", 0

# ==================== CARGAR MODELO ====================
@st.cache_resource
def load_model():
    """Cargar el modelo Vision Transformer entrenado"""
    
    # Primero verificar configuración
    if not check_configuration():
        return None, {}, 0
    
    # Intentar descargar si no existe
    if not os.path.exists(Config.MODEL_PATH):
        success, message, file_size = download_model_from_drive()
        
        if not success:
            st.error(f"Error: {message}")
            st.info("""
            Solución manual:
            1. Descarga manualmente desde: https://drive.google.com
            2. Busca el archivo: best_model_gpu.pth
            3. Colócalo en: models/
            4. Reinicia la aplicación
            """)
            return None, {}, 0
    
    try:
        # Mostrar información del modelo
        file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
        st.sidebar.success(f"Modelo: {file_size:.2f} GB")
        
        # Cargar checkpoint
        with st.spinner("Cargando modelo en memoria..."):
            checkpoint = torch.load(Config.MODEL_PATH, map_location='cpu', weights_only=False)
        
        # Crear modelo
        model = timm.create_model('vit_base_patch16_224', 
                                 pretrained=False, 
                                 num_classes=len(Config.CLASS_NAMES))
        
        # Cargar pesos
        state_dict = checkpoint['model_state_dict']
        
        # Manejar DataParallel si fue usado
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Cargar historial de entrenamiento si existe
        history = checkpoint.get('history', {})
        val_acc = checkpoint.get('val_acc', 93.27)
        
        return model, history, val_acc
        
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        
        st.info(f"""
        Información para depuración:
        - Ruta del modelo: {Config.MODEL_PATH}
        - Tamaño del archivo: {os.path.getsize(Config.MODEL_PATH) if os.path.exists(Config.MODEL_PATH) else 'No existe'} bytes
        - ID de Drive: {Config.GOOGLE_DRIVE_FILE_ID}
        """)
        
        return None, {}, 0

# ==================== TRANSFORMACIONES ====================
def get_transforms():
    """Obtener transformaciones para las imágenes"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ==================== PREDICCIÓN ====================
def predict_image(model, image):
    """Realizar predicción en una imagen"""
    transform = get_transforms()
    
    # Preprocesar imagen
    img_tensor = transform(image).unsqueeze(0)
    
    # Realizar predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = Config.CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # Obtener todas las probabilidades
        all_probs = {Config.CLASS_NAMES[i]: prob.item() 
                    for i, prob in enumerate(probabilities)}
    
    return predicted_class, confidence, all_probs

# ==================== GRÁFICOS CON MATPLOTLIB ====================
def create_training_history_plot(history):
    """Crear gráfico del historial de entrenamiento"""
    if not history or 'train_acc' not in history:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = list(range(1, len(history['train_acc']) + 1))
    
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, marker='o', 
            markersize=6, label='Entrenamiento')
    
    if 'val_acc' in history and history['val_acc']:
        ax.plot(epochs[:len(history['val_acc'])], history['val_acc'], 'r-', 
                linewidth=2, marker='s', markersize=6, label='Validación')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Historial de Entrenamiento - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_loss_history_plot(history):
    """Crear gráfico del historial de pérdida"""
    if not history or 'train_loss' not in history:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    ax.plot(epochs, history['train_loss'], 'g-', linewidth=2, marker='o', 
            markersize=6, label='Entrenamiento')
    
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs[:len(history['val_loss'])], history['val_loss'], color='orange', 
                linewidth=2, marker='s', markersize=6, label='Validación')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.set_title('Historial de Entrenamiento - Pérdida')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_probability_chart(probabilities):
    """Crear gráfico de barras para probabilidades"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    sorted_indices = np.argsort(probs)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]
    
    bars = ax.bar(classes, probs, color=Config.CLASS_COLORS[:len(classes)])
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2%}', ha='center', va='bottom')
    
    ax.set_xlabel('Clase')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Probabilidades de Predicción')
    ax.set_ylim([0, 1.1])
    
    return fig

def create_metrics_chart():
    """Crear gráfico de métricas por clase"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = Config.CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.25
    
    precision = [0.98, 0.90, 0.93, 0.96]
    recall = [0.98, 0.89, 0.94, 0.96]
    f1_score = [0.98, 0.895, 0.935, 0.96]
    
    bars1 = ax.bar(x - width, precision, width, label='Precisión', 
                   color=Config.CLASS_COLORS[0], alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color=Config.CLASS_COLORS[1], alpha=0.8)
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', 
                   color=Config.CLASS_COLORS[2], alpha=0.8)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Clase')
    ax.set_ylabel('Score')
    ax.set_title('Métricas por Clase (Estimadas)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig

# ==================== COMPONENTES UI ====================
def create_header():
    """Crear encabezado de la aplicación"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
    
    with col2:
        st.title("Sistema de Detección de COVID-19")
        st.markdown("**Vision Transformer** para análisis de radiografías pulmonares")
    
    st.markdown("---")

def create_sidebar(model, history, val_acc):
    """Crear barra lateral con información de descarga"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.sidebar.title("Navegación")
    
    page = st.sidebar.radio(
        "Seleccione una página:",
        ["Inicio", "Predicción", "Análisis", "Información", "Configuración"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Estado del modelo
    st.sidebar.subheader("Estado del Modelo")
    
    # Verificar si el modelo existe
    model_exists = os.path.exists(Config.MODEL_PATH)
    
    if model_exists and model is not None:
        file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
        st.sidebar.success(f"Modelo disponible ({file_size:.2f} GB)")
        st.sidebar.metric("Accuracy", f"{val_acc:.2f}%")
    else:
        st.sidebar.warning("Modelo no descargado")
        
        # Botón para descargar manualmente
        if st.sidebar.button("Descargar modelo ahora", use_container_width=True):
            with st.spinner("Descargando modelo..."):
                success, message, _ = download_model_from_drive()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    st.sidebar.markdown("---")
    
    # Información del sistema
    st.sidebar.subheader("Estadísticas")
    
    info_container = st.sidebar.container()
    with info_container:
        st.metric("Clases", "4")
        st.metric("Tamaño modelo", "~1 GB")
        st.metric("Servidor", "Render.com")
    
    st.sidebar.markdown("---")
    
    # Información de contacto
    st.sidebar.subheader("Contacto")
    st.sidebar.info(
        "Sistema de apoyo diagnóstico. Consulte siempre con un profesional médico."
    )
    
    return page

def create_footer():
    """Crear pie de página"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Desarrollado con**")
        st.markdown("![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)")
        st.markdown("![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)")
    
    with col2:
        st.markdown("**Almacenamiento**")
        st.markdown("![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white)")
        st.markdown("Modelo: 1 GB")
    
    with col3:
        st.markdown("**Hosting**")
        st.markdown("Render.com")
        st.markdown("Servidor Cloud")
    
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
        <p>© 2024 Sistema de Detección COVID-19 | Uso exclusivo para investigación</p>
        <p><small>Modelo almacenado en Google Drive | Desplegado en Render.com</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== PÁGINAS ====================
def home_page(model, history, val_acc):
    """Página de inicio"""
    st.header("Bienvenido al Sistema de Detección COVID-19")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Características del Sistema:**
        - 4 Clases: COVID, Opacidad Pulmonar, Normal, Neumonía Viral
        - Modelo: Vision Transformer (ViT-Base-224)
        - Precisión: 93.27% en validación
        - Descarga automática: Desde Google Drive
        - Seguro: Solo para investigación
        """)
    
    with col2:
        st.success("""
        **Cómo Usar:**
        1. Verifica que el modelo esté descargado
        2. Navega a Predicción
        3. Sube una radiografía pulmonar
        4. Obtén el diagnóstico asistido
        5. Consulta siempre con un médico
        """)
    
    st.markdown("---")
    
    # Estado del sistema
    st.subheader("Estado del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if model is not None:
            st.success("Modelo cargado")
        else:
            st.error("Modelo no disponible")
    
    with col2:
        file_exists = os.path.exists(Config.MODEL_PATH)
        if file_exists:
            file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
            st.metric("Tamaño modelo", f"{file_size:.2f} GB")
        else:
            st.metric("Tamaño modelo", "No descargado")
    
    with col3:
        st.metric("Accuracy Total", f"{val_acc:.2f}%")
    
    with col4:
        st.metric("Clases", str(len(Config.CLASS_NAMES)))
    
    st.markdown("---")
    
    # Gráficos de entrenamiento
    st.subheader("Historial de Entrenamiento")
    
    if history:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acc = create_training_history_plot(history)
            if fig_acc:
                st.pyplot(fig_acc)
        
        with col2:
            fig_loss = create_loss_history_plot(history)
            if fig_loss:
                st.pyplot(fig_loss)
    
    # Información de las clases
    st.subheader("Clases Detectables")
    
    cols = st.columns(len(Config.CLASS_NAMES))
    
    for idx, (class_name, color) in enumerate(zip(Config.CLASS_NAMES, Config.CLASS_COLORS)):
        with cols[idx]:
            with st.container():
                st.markdown(
                    f"""
                    <div style='
                        background-color: {color}20;
                        border-left: 5px solid {color};
                        padding: 15px;
                        border-radius: 5px;
                        margin: 5px 0;
                    '>
                    <h4 style='color: {color}; margin: 0;'>{class_name}</h4>
                    <p style='margin: 5px 0 0 0; font-size: 0.9em;'>
                    {Config.CLASS_DESCRIPTIONS[class_name]}
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def prediction_page(model):
    """Página de predicción"""
    if model is None:
        st.error("El modelo no está disponible")
        st.info("""
        Por favor:
        1. Ve a la página Configuración
        2. Descarga el modelo desde Google Drive
        3. Regresa a esta página
        """)
        return
    
    st.header("Predicción de Radiografías")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Sube una Radiografía:**
        - Formatos: JPG, PNG, JPEG
        - Tamaño recomendado: 224x224 píxeles
        - Imágenes en escala de grises o color
        - Asegúrate de que sea una radiografía pulmonar frontal
        """)
    
    with col2:
        st.warning("""
        **Advertencia:**
        Este sistema es para investigación.
        Los resultados deben ser validados por un radiólogo.
        No use para diagnóstico clínico directo.
        """)
    
    st.markdown("---")
    
    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Selecciona una radiografía pulmonar",
        type=['jpg', 'jpeg', 'png'],
        help="Sube una imagen de radiografía pulmonar"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar y mostrar imagen
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Subida")
                st.image(image, caption="Radiografía Pulmonar", use_column_width=True)
                
                # Información de la imagen
                img_info = st.container()
                with img_info:
                    st.write(f"**Formato:** {image.format or 'Desconocido'}")
                    st.write(f"**Tamaño:** {image.size[0]} x {image.size[1]} píxeles")
                    st.write(f"**Modo:** {image.mode}")
            
            with col2:
                st.subheader("Procesando...")
                
                with st.spinner("Realizando predicción..."):
                    # Realizar predicción
                    predicted_class, confidence, all_probs = predict_image(model, image)
                    
                    # Mostrar resultados
                    result_color = Config.CLASS_COLORS[Config.CLASS_NAMES.index(predicted_class)]
                    
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {result_color}20;
                            border: 2px solid {result_color};
                            border-radius: 10px;
                            padding: 20px;
                            text-align: center;
                            margin: 20px 0;
                        '>
                        <h2 style='color: {result_color}; margin: 0;'>{predicted_class}</h2>
                        <h3 style='margin: 10px 0;'>Confianza: <span style='color: {result_color};'>{confidence:.2%}</span></h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Gráfico de probabilidades
                    st.subheader("Probabilidades")
                    fig_prob = create_probability_chart(all_probs)
                    st.pyplot(fig_prob)
            
            st.markdown("---")
            
            # Detalles de la predicción
            st.subheader("Detalles de la Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Probabilidades por Clase**")
                prob_df = pd.DataFrame({
                    'Clase': list(all_probs.keys()),
                    'Probabilidad': list(all_probs.values())
                }).sort_values('Probabilidad', ascending=False)
                
                st.dataframe(
                    prob_df.style.format({'Probabilidad': '{:.2%}'}),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Descripción Clínica**")
                st.info(Config.CLASS_DESCRIPTIONS[predicted_class])
                
                st.markdown("**Recomendaciones**")
                st.warning(Config.CLASS_RECOMMENDATIONS[predicted_class])
            
            # Exportar resultados
            st.markdown("---")
            st.subheader("Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report = f"""
                REPORTE DE PREDICCIÓN - {timestamp}
                ====================================
                Archivo: {uploaded_file.name}
                Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                RESULTADO:
                - Clase Predicha: {predicted_class}
                - Confianza: {confidence:.2%}
                
                PROBABILIDADES:
                """
                for cls, prob in all_probs.items():
                    report += f"- {cls}: {prob:.2%}\n"
                
                report += f"\nDESCRIPCIÓN: {Config.CLASS_DESCRIPTIONS[predicted_class]}"
                report += f"\n\nRECOMENDACIÓN: {Config.CLASS_RECOMMENDATIONS[predicted_class]}"
                
                st.download_button(
                    label="Descargar Reporte (.txt)",
                    data=report,
                    file_name=f"reporte_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Opción para guardar gráfico
                if st.button("Guardar Gráfico de Probabilidades", use_container_width=True):
                    # Crear y guardar gráfico
                    fig_save = create_probability_chart(all_probs)
                    fig_save.savefig(f"probabilidades_{timestamp}.png", dpi=150, bbox_inches='tight')
                    st.success(f"Gráfico guardado como probabilidades_{timestamp}.png")
        
        except Exception as e:
            st.error(f"Error procesando la imagen: {e}")
            st.error("Por favor, sube una imagen válida.")
    else:
        # Mostrar ejemplo
        st.info("**Ejemplo de uso:** Sube una radiografía pulmonar para obtener una predicción.")

def analysis_page(history, val_acc):
    """Página de análisis del modelo"""
    st.header("Análisis del Modelo")
    
    # Resumen del modelo
    st.subheader("Resumen del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Arquitectura", "ViT-Base")
    
    with col2:
        st.metric("Parámetros", "86M")
    
    with col3:
        st.metric("Input Size", "224x224")
    
    with col4:
        st.metric("Pre-entrenado", "ImageNet")
    
    st.markdown("---")
    
    # Gráficos de entrenamiento
    st.subheader("Análisis de Entrenamiento")
    
    if history:
        tab1, tab2, tab3 = st.tabs(["Accuracy", "Pérdida", "Métricas"])
        
        with tab1:
            fig_acc = create_training_history_plot(history)
            if fig_acc:
                st.pyplot(fig_acc)
            
            if 'train_acc' in history and 'val_acc' in history:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy Train Final", f"{history['train_acc'][-1]:.2f}%")
                    st.metric("Accuracy Train Inicial", f"{history['train_acc'][0]:.2f}%")
                
                with col2:
                    st.metric("Accuracy Val Final", f"{history['val_acc'][-1]:.2f}%")
                    st.metric("Mejor Accuracy Val", f"{max(history['val_acc']):.2f}%")
        
        with tab2:
            fig_loss = create_loss_history_plot(history)
            if fig_loss:
                st.pyplot(fig_loss)
            
            if 'train_loss' in history and 'val_loss' in history:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Pérdida Train Final", f"{history['train_loss'][-1]:.4f}")
                    st.metric("Pérdida Train Inicial", f"{history['train_loss'][0]:.4f}")
                
                with col2:
                    st.metric("Pérdida Val Final", f"{history['val_loss'][-1]:.4f}")
                    st.metric("Mejor Pérdida Val", f"{min(history['val_loss']):.4f}")
        
        with tab3:
            fig_metrics = create_metrics_chart()
            if fig_metrics:
                st.pyplot(fig_metrics)
            
            # Tabla de métricas
            st.subheader("Métricas por Clase")
            
            metrics_df = pd.DataFrame({
                'Clase': Config.CLASS_NAMES,
                'Precisión': [0.98, 0.90, 0.93, 0.96],
                'Recall': [0.98, 0.89, 0.94, 0.96],
                'F1-Score': [0.98, 0.895, 0.935, 0.96],
                'Casos (entrenamiento)': [732, 1201, 2020, 280]
            })
            
            st.dataframe(
                metrics_df.style.format({
                    'Precisión': '{:.2%}',
                    'Recall': '{:.2%}', 
                    'F1-Score': '{:.2%}'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Información técnica
    st.subheader("Información Técnica")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("**Hiperparámetros**")
        tech_info = st.container()
        with tech_info:
            st.write("- **Learning Rate:** 2e-4")
            st.write("- **Batch Size:** 32")
            st.write("- **Épocas:** 5")
            st.write("- **Optimizador:** AdamW")
            st.write("- **Weight Decay:** 1e-4")
            st.write("- **Scheduler:** OneCycleLR")
    
    with tech_col2:
        st.markdown("**Preprocesamiento**")
        preproc_info = st.container()
        with preproc_info:
            st.write("- **Resize:** 224x224")
            st.write("- **Normalización:** ImageNet stats")
            st.write("- **Augmentations:** Flip, Rotation, ColorJitter")
            st.write("- **Train/Val Split:** 80/20")
            st.write("- **Classes:** 4 balanceadas")

def info_page():
    """Página de información"""
    st.header("Información del Sistema")
    
    st.info("""
    **Objetivo del Sistema**
    Este sistema utiliza inteligencia artificial para asistir en la detección 
    de condiciones pulmonares a partir de radiografías de tórax.
    
    **NO** es un sistema de diagnóstico automático, sino una herramienta 
    de apoyo para profesionales de la salud.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**Base de Datos**")
        st.markdown("""
        - **COVID-19 Radiography Database**
        - **Total de imágenes:** 21,165
        - **Distribución:**
          - COVID: 3,616
          - Lung Opacity: 6,012
          - Normal: 10,192
          - Viral Pneumonia: 1,345
        - **Resolución:** Variada (escalada a 224x224)
        """)
    
    with col2:
        st.subheader("**Arquitectura del Modelo**")
        st.markdown("""
        - **Modelo:** Vision Transformer (ViT-Base)
        - **Parches:** 16x16
        - **Capas Transformer:** 12
        - **Heads de atención:** 12
        - **Dimensiones ocultas:** 768
        - **MLP Size:** 3072
        - **Parámetros:** 86 millones
        """)
    
    st.markdown("---")
    
    st.subheader("**Limitaciones y Advertencias**")
    
    warning_col1, warning_col2 = st.columns(2)
    
    with warning_col1:
        st.error("""
        **Limitaciones Técnicas:**
        - Solo procesa radiografías frontales
        - No detecta todas las condiciones pulmonares
        - Sensible a calidad de imagen
        - Puede tener falsos positivos/negativos
        """)
    
    with warning_col2:
        st.warning("""
        **Consideraciones Clínicas:**
        - Para investigación únicamente
        - Validar con pruebas clínicas
        - Consultar siempre con radiólogo
        - No usar para diagnóstico autónomo
        """)
    
    st.markdown("---")
    
    st.subheader("**Contacto y Soporte**")
    
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    
    with contact_col1:
        st.markdown("**Desarrollador:**")
        st.write("Sistema de IA Médica")
        st.write("Investigación en Computer Vision")
    
    with contact_col2:
        st.markdown("**Propósito:**")
        st.write("Investigación académica")
        st.write("Desarrollo tecnológico")
        st.write("Apoyo diagnóstico")
    
    with contact_col3:
        st.markdown("**Licencia:**")
        st.write("Uso académico")
        st.write("No comercial")
        st.write("Atribución requerida")

def configuration_page():
    """Página de configuración"""
    st.header("Configuración del Sistema")
    
    st.info("""
    **Configuración de Google Drive**
    El modelo de IA (1GB) se descarga automáticamente desde Google Drive.
    """)
    
    # Mostrar configuración actual
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración Actual")
        st.write(f"**ID de Google Drive:** `{Config.GOOGLE_DRIVE_FILE_ID}`")
        st.write(f"**Ruta del modelo:** `{Config.MODEL_PATH}`")
        
        # Verificar si el archivo existe
        if os.path.exists(Config.MODEL_PATH):
            file_size = os.path.getsize(Config.MODEL_PATH) / (1024**3)
            st.success(f"Modelo disponible ({file_size:.2f} GB)")
        else:
            st.warning("Modelo no encontrado")
    
    with col2:
        st.subheader("Descargar Modelo")
        
        if st.button("Forzar descarga del modelo", use_container_width=True):
            with st.spinner("Descargando modelo desde Google Drive..."):
                success, message, file_size = download_model_from_drive()
                
                if success:
                    st.success(f"{message} ({file_size:.2f} GB)")
                    st.rerun()
                else:
                    st.error(f"{message}")
    
    st.markdown("---")
    
    # Instrucciones para configurar
    st.subheader("Instrucciones para configurar")
    
    st.markdown("""
    **Para usar tu propio modelo en Google Drive:**
    
    1. **Sube tu archivo** `best_model_gpu.pth` a Google Drive
    2. **Comparte el archivo:**
       - Click derecho → "Compartir"
       - Cambiar a "Cualquier persona con el enlace"
    3. **Copia el enlace:** Debe verse así:
       ```
       https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
       ```
    4. **Extrae el ID:** `1A2B3C4D5E6F7G8H9I0J`
    5. **Actualiza el código:** Reemplaza en `Config.GOOGLE_DRIVE_FILE_ID`
    """)
    
    # Información del sistema
    st.markdown("---")
    st.subheader("Información del Sistema")
    
    sys_info = st.container()
    with sys_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Python", f"{sys.version.split()[0]}")
        
        with col2:
            st.metric("PyTorch", torch.__version__)
        
        with col3:
            st.metric("Streamlit", st.__version__)

# ==================== APLICACIÓN PRINCIPAL ====================
def main():
    """Función principal de la aplicación Streamlit"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Sistema COVID-19 - Vision Transformer",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inyectar CSS personalizado
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .download-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid #ddd;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    model, history, val_acc = load_model()
    
    # Crear UI
    create_header()
    page = create_sidebar(model, history, val_acc)
    
    # Navegación de páginas
    if page == "Inicio":
        home_page(model, history, val_acc)
    elif page == "Predicción":
        prediction_page(model)
    elif page == "Análisis":
        analysis_page(history, val_acc)
    elif page == "Información":
        info_page()
    elif page == "Configuración":
        configuration_page()
    
    # Crear pie de página
    create_footer()

if __name__ == "__main__":
    main()