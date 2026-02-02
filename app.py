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

# =========================================================================
# 1. IMPORTATION ET SÉCURISATION DES DÉPENDANCES
# =========================================================================
try:
    # Import du fichier local 'model.py' qui contient l'architecture du réseau neuronal
    from model import get_model_instance_segmentation
    MODEL_AVAILABLE = True
except ImportError:
    # Filet de sécurité : Affiche une erreur visuelle si le fichier manque
    st.error(" ERREUR CRITIQUE : Le fichier 'model.py' est introuvable. Vérifiez l'arborescence.")
    MODEL_AVAILABLE = False

# =========================================================================
# 2. CONFIGURATION FRONT-END (INTERFACE UTILISATEUR)
# =========================================================================
# Configuration de la page en mode "Large" pour optimiser l'espace pour les images
st.set_page_config(layout="wide", page_title="Détecteur d'avions militaires", page_icon="✈️")

# Injection de CSS (Feuille de style) pour forcer un rendu professionnel
# On uniformise les polices et on crée des encadrés propres pour les métriques.
st.markdown("""
    <style>
    /* Force le fond blanc pour un rendu type "Rapport" */
    .stApp { background-color: #ffffff; color: #000000; }
    
    /* Styles des titres H1 et H2 */
    h1 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
    h2, h3 { color: #333333; }
    
    /* Design des cartes de métriques (KPI) */
    .stMetric { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; text-align: center; }
    div[data-testid="stMetricValue"] { color: #0056b3; font-weight: bold; font-size: 1.5rem; }
    
    /* Bouton d'action vert (Action positive) */
    .stButton>button { background-color: #28a745; color: white; border-radius: 5px; border: none; }
    </style>
    """, unsafe_allow_html=True)

# Liste des 20 classes d'avions + le fond (Background)
# Ces classes correspondent exactement à celles du dataset Pascal VOC utilisé à l'entraînement.
CLASSES = ['Background', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

# =========================================================================
# 3. FONCTIONS SYSTÈME 
# =========================================================================

# @st.cache_resource est ESSENTIEL : 
# Il charge le modèle une seule fois en mémoire (RAM/VRAM) au démarrage.
# Sans ça, le modèle serait rechargé à chaque clic, rendant l'app très lente.
@st.cache_resource
def load_system(model_path):
    """
    Charge le modèle Faster R-CNN, initialise l'architecture et charge les poids (.pth).
    Détecte automatiquement si un GPU (CUDA) est disponible pour accélérer l'inférence.
    """
    if not MODEL_AVAILABLE: return None, False
    
    # Sélection dynamique du processeur : Carte Graphique (GPU) ou Processeur (CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        # 1. Instanciation de l'architecture vide (21 classes)
        model = get_model_instance_segmentation(21)
        
        # 2. Chargement des poids entraînés (State Dict)
        if os.path.exists(model_path):
            # 'map_location' gère la compatibilité si le modèle a été entraîné sur GPU mais tourne sur CPU
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # 3. Envoi du modèle vers le périphérique de calcul (GPU/CPU)
            model.to(device)
            
            # 4. Mode Évaluation : Fige les couches comme le Dropout ou BatchNormalization
            model.eval() 
            return model, True
        else:
            return None, False
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return None, False

def run_inference_pytorch(model, img_input, conf_threshold, nms_threshold):
    """
    Exécute la détection d'objets sur une image.
    Entrée : Image brute (Numpy ou PIL).
    Sortie : Liste de dictionnaires contenant boîtes, labels et scores.
    """
    # Récupération du device (CPU ou GPU) sur lequel est le modèle
    device = next(model.parameters()).device
    
    # --- A. PRÉPARATION DE L'IMAGE (PIPELINE) ---
    # Conversion OpenCV (BGR) vers PIL (RGB) si nécessaire
    if isinstance(img_input, np.ndarray):
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = img_input

    # Transformation en Tenseur PyTorch :
    # 1. Convertit les pixels [0, 255] en float [0.0, 1.0]
    # 2. Change l'ordre des dimensions (H, W, C) -> (C, H, W)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).to(device)
    
    # Ajout d'une dimension de "batch" (N, C, H, W) car le modèle attend un lot d'images
    img_tensor = img_tensor.unsqueeze(0) 
    
    # --- B. INFÉRENCE (FORWARD PASS) ---
    # 'torch.no_grad()' désactive le calcul des gradients pour économiser la mémoire (pas d'apprentissage ici)
    with torch.no_grad():
        prediction = model(img_tensor)
        
    # Extraction des résultats bruts du premier élément du batch
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']
    
    # --- C. POST-TRAITEMENT (NMS) ---
    # Non-Maximum Suppression : Algorithme qui fusionne les boîtes qui se chevauchent trop.
    # Si deux boîtes détectent le même objet, on garde celle avec le meilleur score.
    keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
    
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # Transfert des données du GPU vers le CPU pour traitement Python standard
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # --- D. FILTRAGE FINAL ---
    results = []
    for i in range(len(boxes)):
        # On ne garde que les détections supérieures au seuil de confiance utilisateur
        if scores[i] > conf_threshold:
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1) # Calcul de la surface (pour info)
            results.append({
                "box": boxes[i],
                "score": scores[i],
                "label": CLASSES[labels[i]],
                "area": area
            })
    return results

def process_signal(img, bri, con, gam, sha):
    """
    Traitement du Signal Numérique (DSP) via OpenCV.
    Permet d'améliorer la visibilité des caractéristiques pour l'IA ou l'œil humain.
    """
    # 1. Transformation Linéaire (Luminosité & Contraste)
    # Formule : pixel_out = alpha * pixel_in + beta
    img = cv2.convertScaleAbs(img, alpha=con, beta=bri)
    
    # 2. Correction Gamma (Non-linéaire)
    # Permet d'éclaircir les zones sombres sans brûler les zones claires
    if gam != 1.0:
        invGamma = 1.0 / gam
        # Création d'une table de correspondance (Look-Up Table) pour rapidité
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    
    # 3. Filtre de Netteté (Sharpening)
    # Convolution avec un noyau laplacien modifié pour accentuer les arêtes (Edges)
    if sha > 0:
        kernel = np.array([[-1,-1,-1], [-1, 9+(sha/5.0), -1], [-1,-1,-1]])
        kernel = kernel / np.sum(kernel) # Normalisation pour ne pas changer la luminosité moyenne
        img = cv2.filter2D(img, -1, kernel)
    return img

# =========================================================================
# 4. ORCHESTRATION DE L'INTERFACE (GUI)
# =========================================================================

st.title("Détecteur d'avions")
st.markdown("**Architecture :** PyTorch (Faster R-CNN)")

# --- BARRE LATÉRALE (CONTROLES UTILISATEUR) ---
st.sidebar.header("🎛️ Pré-traitement du signal")
# Les sliders renvoient directement la valeur choisie par l'utilisateur
bri = st.sidebar.slider("Luminosité", -50, 50, 0)
con = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0)
gam = st.sidebar.slider("Gamma", 0.5, 2.5, 1.0)
sha = st.sidebar.slider("Netteté (Edges)", 0, 10, 0)

st.sidebar.markdown("---")
st.sidebar.header("🧠 Configuration IA")

# Construction robuste du chemin fichier (compatible Windows/Linux/Docker)
default_path = os.path.join("models", "faster_rcnn_avions_V2.pth")
model_path = st.sidebar.text_input("Chemin Modèle (.pth)", default_path)

# Paramètres critiques de l'inférence
conf_thresh = st.sidebar.slider("Seuil Détection (Précision)", 0.0, 1.0, 0.50)
nms_thresh = st.sidebar.slider("Seuil Doublons (NMS)", 0.0, 1.0, 0.3)

# Tentative de chargement du système
model, status = load_system(model_path)

if status:
    st.sidebar.success("✅ SYSTÈME EN LIGNE")
else:
    st.sidebar.error(f"⚠️ HORS LIGNE : Modèle introuvable à '{model_path}'")

# --- ZONE PRINCIPALE (UPLOAD) ---
uploaded_files = st.file_uploader("IMPORTER UNE IMAGE SATELLITE", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    # Sélecteur d'image si plusieurs fichiers sont chargés
    if len(uploaded_files) > 1:
        st.info(f"📁 {len(uploaded_files)} fichiers chargés.")
        file_map = {f.name: f for f in uploaded_files}
        selected_name = st.selectbox("Sélectionner une image à visualiser :", list(file_map.keys()))
        selected_file = file_map[selected_name]
    else:
        selected_file = uploaded_files[0]
        selected_name = selected_file.name

    # 1. LECTURE DU FICHIER BINAIRE
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    
    # CRITIQUE : On rembobine le pointeur du fichier à 0. 
    # Sinon, lors d'une prochaine lecture (ex: scan global), le fichier semblerait vide.
    selected_file.seek(0)
    
    # Décodage OpenCV
    img_raw = cv2.imdecode(file_bytes, 1)
    
    # 2. APPLICATION DU DSP (Traitement Signal)
    img_processed = process_signal(img_raw, bri, con, gam, sha)
    
    # 3. BOUCLE DE DÉTECTION
    detections = []
    output_img = img_processed.copy()
    
    # Début du chronomètre pour mesurer la latence
    t_start = time.perf_counter()
    
    if status:
        # Appel de l'IA
        detections = run_inference_pytorch(model, img_processed, conf_thresh, nms_thresh)
        
        # Dessin des résultats (Bounding Boxes)
        for res in detections:
            box = res['box']
            x1, y1, x2, y2 = map(int, box) # Coordonnées en entiers pour OpenCV
            color = (0, 255, 0) # Vert pur
            
            # Dessin du rectangle
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            
            # Ajout du label textuel
            label_txt = f"{res['label']} P:{res['score']:.2f}"
            cv2.putText(output_img, label_txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Fin du chronomètre
    latency_ms = (time.perf_counter() - t_start) * 1000

    # 4. AFFICHAGE DES IMAGES
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Entrée (Signal Brut)")
        # Conversion BGR (OpenCV) -> RGB (Écran)
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.caption(f"Sortie (Détection IA)")
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Temps de traitement", f"{latency_ms:.1f} ms")

    # =========================================================
    # 5. RAPPORT ET ANALYSE
    # =========================================================
    st.markdown("---")
    st.subheader("📋 Résultats")

    tab_single, tab_batch = st.tabs(["🔎 Pour une image", "🌍 Pour plusieurs images"])

    # --- ONGLET 1 : ANALYSE UNITAIRE ---
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
            # Création du DataFrame Pandas pour l'affichage tableau
            df_single = pd.DataFrame(data_list)
            
            c_table, c_metric = st.columns([3, 1])
            with c_table:
                st.dataframe(df_single, use_container_width=True)
            with c_metric:
                st.metric("Avions Détectés", len(detections))
                # Moyenne des scores de confiance
                st.metric("Précision Moyenne", f"{pd.Series([d['score'] for d in detections]).mean():.1%}")
        else:
            st.info("Aucun avion détecté sur cette image.")

    # --- ONGLET 2 : ANALYSE PAR LOT (BATCH) ---
    with tab_batch:
        if len(uploaded_files) > 1:
            st.write("Analyse de l'ensemble des fichiers importés.")
            if st.button(f"LANCER LE TRAITEMENT SUR {len(uploaded_files)} FICHIERS"):
                if status:
                    all_detections = []
                    progress_bar = st.progress(0)
                    
                    # Itération sur chaque fichier chargé
                    for i, file in enumerate(uploaded_files):
                        file.seek(0) # Reset pointeur obligatoire !
                        f_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        im = cv2.imdecode(f_bytes, 1)
                        
                        # On applique le même DSP que celui réglé dans la sidebar
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
                            # Enregistrement des "R.A.S" (Rien à Signaler) pour les stats
                            all_detections.append({
                                "Image": file.name,
                                "Classe": "R.A.S",
                                "Précision": 0.0,
                                "Temps (ms)": int(dt),
                                "Coordonnées": "-"
                            })
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Synthèse finale
                    if all_detections:
                        df_all = pd.DataFrame(all_detections)
                        df_display = df_all.sort_values(by="Précision", ascending=False).copy()
                        df_display["Précision"] = df_display["Précision"].apply(lambda x: f"{x:.2%}")
                        
                        st.success(f"Scan terminé : {len(all_detections)} objets traités.")
                        st.dataframe(df_display, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("📊 Répartition des Classes")
                        
                        # # Graphique de répartition
                        df_chart = df_all[df_all['Classe'] != "R.A.S"]
                        if not df_chart.empty:
                            st.bar_chart(df_chart['Classe'].value_counts())
                        else:
                            st.info("Rien à signaler sur les graphiques.")
                    else:
                        st.warning("Aucune détection.")
                else:
                    st.error("Modèle non chargé.")
        else:
            st.info("Chargez plusieurs images pour activer le Scan Global.")
else:
    # État initial (attente d'upload)
    st.info("En attente d'images satellite...")