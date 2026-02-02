import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
import pandas as pd
import os
import time
from PIL import Image

# --- 1. IMPORT DU CERVEAU (MODEL.PY) ---
try:
    from model import get_model_instance_segmentation
    MODEL_AVAILABLE = True
except ImportError:
    st.error(" ERREUR : Le fichier 'model.py' est introuvable. Assurez-vous qu'il est dans le même dossier que app.py.")
    MODEL_AVAILABLE = False

# --- 2. CONFIGURATION & DESIGN ---
st.set_page_config(layout="wide", page_title="Détecteur d'avions militaires", page_icon="✈️")

st.markdown("""
    <style>
    /* FOND GLOBAL BLANC */
    .stApp { background-color: #ffffff; color: #000000; }
    h1 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
    h2, h3 { color: #333333; }
    .stMetric { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); text-align: center; }
    div[data-testid="stMetricValue"] { color: #0056b3; font-family: 'Arial', sans-serif; font-weight: bold; font-size: 1.5rem; }
    div[data-testid="stMetricLabel"] { color: #333333; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #f1f3f4; border-radius: 5px; color: #444; border: 1px solid #ddd; }
    .stTabs [aria-selected="true"] { background-color: #0056b3; color: white; border-color: #0056b3; }
    .stButton>button { background-color: #28a745; color: white; font-weight: bold; border: none; width: 100%; border-radius: 5px; }
    .stDataFrame { border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FONCTIONS SYSTÈME & IA ---

CLASSES = ['Background', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

@st.cache_resource
def load_system(model_path):
    """Charge le modèle Pytorch"""
    if not MODEL_AVAILABLE: return None, False
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        model = get_model_instance_segmentation(21)
        # Vérification robuste du fichier
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, True
        else:
            return None, False
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return None, False

def run_inference_pytorch(model, img_input, conf_threshold, nms_threshold):
    device = next(model.parameters()).device
    
    if isinstance(img_input, np.ndarray):
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = img_input

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).to(device)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']
    
    keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    
    results = []
    for i in range(len(boxes)):
        if scores[i] > conf_threshold:
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1)
            results.append({
                "box": boxes[i],
                "score": scores[i],
                "label": CLASSES[labels[i]],
                "area": area
            })
    return results

def process_signal(img, bri, con, gam, sha):
    img = cv2.convertScaleAbs(img, alpha=con, beta=bri)
    if gam != 1.0:
        invGamma = 1.0 / gam
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    if sha > 0:
        kernel = np.array([[-1,-1,-1], [-1, 9+(sha/5.0), -1], [-1,-1,-1]])
        kernel = kernel / np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
    return img

# --- 5. INTERFACE PRINCIPALE ---

st.title("Détecteur d'avions")
st.markdown("**Architecture :** PyTorch (Faster R-CNN) ")

# SIDEBAR - TRAITEMENT
st.sidebar.header("🎛️ Options de pré-traitement image")
bri = st.sidebar.slider("Luminosité", -50, 50, 0)
con = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0)
gam = st.sidebar.slider("Gamma", 0.5, 2.5, 1.0)
sha = st.sidebar.slider("Netteté (Edges)", 0, 10, 0)

st.sidebar.markdown("---")
# SIDEBAR - IA
st.sidebar.header("🧠 Configuration")

#  CHEMIN VERS LE DOSSIER MODELS 
default_path = os.path.join("models", "faster_rcnn_avions_V2.pth")
model_path = st.sidebar.text_input("Chemin Modèle (.pth)", default_path)

conf_thresh = st.sidebar.slider("Seuil Détection (Précision)", 0.0, 1.0, 0.50)
nms_thresh = st.sidebar.slider("Seuil Doublons (NMS)", 0.0, 1.0, 0.3)

model, status = load_system(model_path)
if status:
    st.sidebar.success("✅ SYSTÈME EN LIGNE")
else:
    st.sidebar.error(f"⚠️ HORS LIGNE : Impossible de trouver '{model_path}'")

# ==========================================
# Partie traitement 
# ==========================================

uploaded_files = st.file_uploader("IMPORTER UNE IMAGE SATELLITE", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 1:
        st.info(f"📁 {len(uploaded_files)} fichiers chargés.")
        file_map = {f.name: f for f in uploaded_files}
        selected_name = st.selectbox("Sélectionner une image à visualiser :", list(file_map.keys()))
        selected_file = file_map[selected_name]
    else:
        selected_file = uploaded_files[0]
        selected_name = selected_file.name

    # 1. TRAITEMENT
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    selected_file.seek(0)
    
    img_raw = cv2.imdecode(file_bytes, 1)
    img_processed = process_signal(img_raw, bri, con, gam, sha)
    
    # 2. INFÉRENCE
    detections = []
    output_img = img_processed.copy()
    t_start = time.perf_counter()
    
    if status:
        detections = run_inference_pytorch(model, img_processed, conf_thresh, nms_thresh)
        for res in detections:
            box = res['box']
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{res['label']} P:{res['score']:.2f}"
            cv2.putText(output_img, label_txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    latency_ms = (time.perf_counter() - t_start) * 1000

    # 3. AFFICHAGE
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Entrée")
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.caption(f"Sortie")
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Temps de traitement", f"{latency_ms:.1f} ms")

    # =========================================================
    # RÉSULTATS
    # =========================================================
    st.markdown("---")
    st.subheader("📋 Résultats")

    tab_single, tab_batch = st.tabs(["🔎 Pour une image", "🌍 Pour plusieurs images"])

    with tab_single:
        if detections:
            data_list = []
            for d in detections:
                x1, y1, x2, y2 = map(int, d['box'])
                data_list.append({
                    "Image": selected_name,
                    "Classe": d['label'],
                    "Précision": f"{d['score']:.2%}",
                    "Coordonnées": f"[{x1}, {y1}, {x2}, {y2}]"
                })
            df_single = pd.DataFrame(data_list)
            
            c_table, c_metric = st.columns([3, 1])
            with c_table:
                st.dataframe(df_single, use_container_width=True)
            with c_metric:
                st.metric("Avions Détectés", len(detections))
                st.metric("Précision Moyenne", f"{pd.Series([d['score'] for d in detections]).mean():.1%}")
        else:
            st.info("Aucun avion détectée sur cette image.")

    with tab_batch:
        if len(uploaded_files) > 1:
            st.write("Analyse  de l'ensemble des fichiers importés.")
            if st.button(f"LANCER LE TRAITEMENT SUR {len(uploaded_files)} FICHIERS"):
                if status:
                    all_detections = []
                    progress_bar = st.progress(0)
                    for i, file in enumerate(uploaded_files):
                        file.seek(0) 
                        f_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        im = cv2.imdecode(f_bytes, 1)
                        im_proc = process_signal(im, bri, con, gam, sha)
                        t0 = time.perf_counter()
                        dets = run_inference_pytorch(model, im_proc, conf_thresh, nms_thresh)
                        dt = (time.perf_counter() - t0) * 1000
                        if dets:
                            for d in dets:
                                x1, y1, x2, y2 = map(int, d['box'])
                                all_detections.append({
                                    "Image": file.name,
                                    "Classe": d['label'],
                                    "Précision": d['score'],
                                    "Temps de traitement (ms)": int(dt),
                                    "Coordonnées": f"[{x1}, {y1}, {x2}, {y2}]"
                                })
                        else:
                            all_detections.append({
                                "Image": file.name,
                                "Classe": "R.A.S",
                                "Précision": 0.0,
                                "Temps (ms)": int(dt),
                                "Coordonnées": "-"
                            })
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if all_detections:
                        df_all = pd.DataFrame(all_detections)
                        df_display = df_all.sort_values(by="Précision", ascending=False).copy()
                        df_display["Précision"] = df_display["Précision"].apply(lambda x: f"{x:.2%}")
                        st.success(f"Scan terminé : {len(all_detections)} objets traités.")
                        st.dataframe(df_display, use_container_width=True)
                        st.markdown("---")
                        st.subheader("📊 Répartition des Classes")
                        df_chart = df_all[df_all['Classe'] != "R.A.S"]
                        if not df_chart.empty:
                            st.bar_chart(df_chart['Classe'].value_counts())
                        else:
                            st.info("Rien à signaler.")
                    else:
                        st.warning("Aucune détection.")
                else:
                    st.error("Modèle non chargé.")
        else:
            st.info("Chargez plusieurs images pour activer le Scan Global.")
else:
    st.info("En attente d'images satellite...")