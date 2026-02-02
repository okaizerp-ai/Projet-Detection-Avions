"""
app.py - Interface Graphique Web avec Streamlit (PARTIE 2/2 - INTERFACE UTILISATEUR)

Cette partie contient l'interface utilisateur complète avec:
- Sidebar pour les contrôles (prétraitement, configuration IA)
- Zone d'upload d'images
- Visualisation côte à côte (avant/après)
- Tableaux de résultats (une image / batch)
- Graphiques de statistiques
"""

# [IMPORTS ET FONCTIONS DE LA PARTIE 1 - Voir app_part1.py]

# ========== 5. INTERFACE PRINCIPALE ==========

# ========== EN-TÊTE DE L'APPLICATION ==========
st.title("Détecteur d'avions")  # Titre principal de la page
st.markdown("**Architecture :** PyTorch (Faster R-CNN) ")  # Sous-titre avec l'architecture

# ========== SIDEBAR - SECTION PRÉTRAITEMENT IMAGE ==========
st.sidebar.header("🎛️ Options de pré-traitement image")

# Slider pour ajuster la luminosité
# Plage: -50 à +50, valeur par défaut: 0 (pas de changement)
bri = st.sidebar.slider("Luminosité", -50, 50, 0)

# Slider pour ajuster le contraste
# Plage: 0.5 (faible contraste) à 2.0 (fort contraste), valeur par défaut: 1.0
con = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0)

# Slider pour correction gamma
# Plage: 0.5 (éclaircit) à 2.5 (assombrit), valeur par défaut: 1.0
gam = st.sidebar.slider("Gamma", 0.5, 2.5, 1.0)

# Slider pour amélioration de la netteté
# Plage: 0 (pas d'amélioration) à 10 (très net), valeur par défaut: 0
sha = st.sidebar.slider("Netteté (Edges)", 0, 10, 0)

st.sidebar.markdown("---")  # Ligne de séparation horizontale

# ========== SIDEBAR - SECTION CONFIGURATION IA ==========
st.sidebar.header("🧠 Configuration")

# Champ texte pour spécifier le chemin du modèle
# Valeur par défaut: models/faster_rcnn_avions_V2.pth
default_path = os.path.join("models", "faster_rcnn_avions_V2.pth")
model_path = st.sidebar.text_input("Chemin Modèle (.pth)", default_path)

# Slider pour le seuil de détection (confidence threshold)
# Plage: 0.0 (accepte tout) à 1.0 (très strict), valeur par défaut: 0.50
conf_thresh = st.sidebar.slider("Seuil Détection (Précision)", 0.0, 1.0, 0.50)

# Slider pour le seuil NMS (Non-Maximum Suppression)
# Plage: 0.0 à 1.0, valeur par défaut: 0.3
# Plus bas = élimine plus de doublons, plus haut = garde plus de boîtes
nms_thresh = st.sidebar.slider("Seuil Doublons (NMS)", 0.0, 1.0, 0.3)

# ========== CHARGEMENT DU MODÈLE (AVEC CACHE) ==========
# load_system() est appelée avec le chemin du modèle
# Grâce au décorateur @st.cache_resource, le modèle n'est chargé qu'une seule fois
model, status = load_system(model_path)

# Affichage du statut dans la sidebar
if status:
    # Si le modèle est chargé avec succès, affiche un message vert
    st.sidebar.success("✅ SYSTÈME EN LIGNE")
else:
    # Si le chargement a échoué, affiche un message d'erreur rouge
    st.sidebar.error(f"⚠️ HORS LIGNE : Impossible de trouver '{model_path}'")

# ========== ZONE D'UPLOAD D'IMAGES ==========
# st.file_uploader() crée un bouton pour uploader des fichiers
# type: limite aux formats image
# accept_multiple_files=True: permet de sélectionner plusieurs images en une fois
uploaded_files = st.file_uploader(
    "IMPORTER UNE IMAGE SATELLITE", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

# ========== TRAITEMENT DES IMAGES UPLOADÉES ==========
if uploaded_files:
    # Si l'utilisateur a uploadé au moins une image
    
    # ========== GESTION MULTI-FICHIERS ==========
    if len(uploaded_files) > 1:
        # Si plusieurs images, affiche le nombre total
        st.info(f"🗂 {len(uploaded_files)} fichiers chargés.")
        
        # Création d'un dictionnaire {nom_fichier: objet_fichier}
        file_map = {f.name: f for f in uploaded_files}
        
        # Dropdown pour sélectionner quelle image visualiser
        selected_name = st.selectbox("Sélectionner une image à visualiser :", list(file_map.keys()))
        
        # Récupération de l'objet fichier sélectionné
        selected_file = file_map[selected_name]
    else:
        # Si une seule image, la sélectionne automatiquement
        selected_file = uploaded_files[0]
        selected_name = selected_file.name

    # ========== 1. CHARGEMENT ET PRÉTRAITEMENT DE L'IMAGE ==========
    # Lecture du fichier uploadé en bytes
    file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
    
    # Réinitialisation du curseur du fichier (nécessaire pour relire plus tard)
    selected_file.seek(0)
    
    # Décodage des bytes en image OpenCV (array NumPy BGR)
    img_raw = cv2.imdecode(file_bytes, 1)
    
    # Application des ajustements d'image (luminosité, contraste, gamma, netteté)
    img_processed = process_signal(img_raw, bri, con, gam, sha)
    
    # ========== 2. INFÉRENCE (DÉTECTION) ==========
    detections = []  # Liste qui stockera les détections
    output_img = img_processed.copy()  # Copie de l'image pour dessiner dessus
    
    # Démarrage du chronomètre pour mesurer le temps de traitement
    t_start = time.perf_counter()
    
    # Si le modèle est chargé, effectue l'inférence
    if status:
        # Appel de la fonction d'inférence
        detections = run_inference_pytorch(model, img_processed, conf_thresh, nms_thresh)
        
        # ========== DESSIN DES BOÎTES SUR L'IMAGE ==========
        for res in detections:
            # Extraction de la boîte et conversion en int
            box = res['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Couleur verte (format BGR pour OpenCV)
            color = (0, 255, 0)
            
            # Dessin du rectangle sur l'image
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            
            # Construction du label texte
            label_txt = f"{res['label']} P:{res['score']:.2f}"
            
            # Dessin du texte au-dessus de la boîte
            cv2.putText(output_img, label_txt, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Calcul du temps de traitement en millisecondes
    latency_ms = (time.perf_counter() - t_start) * 1000

    # ========== 3. AFFICHAGE CÔTE À CÔTE ==========
    # st.columns(2) crée 2 colonnes de largeur égale
    c1, c2 = st.columns(2)
    
    # Colonne de gauche : image originale
    with c1:
        st.caption("Entrée")  # Légende
        # Conversion BGR (OpenCV) → RGB (Streamlit)
        # use_container_width=True adapte la largeur à la colonne
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Colonne de droite : image annotée
    with c2:
        st.caption(f"Sortie")  # Légende
        # Affichage de l'image avec les détections dessinées
        st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        # Métrique affichant le temps de traitement
        st.metric("Temps de traitement", f"{latency_ms:.1f} ms")

    # ========== SÉPARATEUR ==========
    st.markdown("---")  # Ligne horizontale
    st.subheader("📋 Résultats")  # Sous-titre

    # ========== ONGLETS DE RÉSULTATS ==========
    # st.tabs() crée des onglets cliquables
    tab_single, tab_batch = st.tabs(["🔎 Pour une image", "🌐 Pour plusieurs images"])

    # ========== ONGLET 1 : RÉSULTATS POUR UNE IMAGE ==========
    with tab_single:
        if detections:
            # Si des avions ont été détectés
            
            # Construction d'un DataFrame Pandas pour affichage en tableau
            data_list = []
            for d in detections:
                # Extraction et conversion des coordonnées en int
                x1, y1, x2, y2 = map(int, d['box'])
                
                # Ajout d'une ligne au tableau
                data_list.append({
                    "Image": selected_name,
                    "Classe": d['label'],
                    "Précision": f"{d['score']:.2%}",  # Format en pourcentage
                    "Coordonnées": f"[{x1}, {y1}, {x2}, {y2}]"
                })
            
            # Conversion de la liste en DataFrame Pandas
            df_single = pd.DataFrame(data_list)
            
            # Affichage en 2 colonnes : tableau + métriques
            c_table, c_metric = st.columns([3, 1])  # Ratio 3:1
            
            with c_table:
                # Affichage du tableau des détections
                st.dataframe(df_single, use_container_width=True)
            
            with c_metric:
                # Métriques récapitulatives
                st.metric("Avions Détectés", len(detections))
                # Calcul de la précision moyenne
                avg_score = pd.Series([d['score'] for d in detections]).mean()
                st.metric("Précision Moyenne", f"{avg_score:.1%}")
        else:
            # Si aucune détection
            st.info("Aucun avion détecté sur cette image.")

    # ========== ONGLET 2 : TRAITEMENT BATCH (PLUSIEURS IMAGES) ==========
    with tab_batch:
        if len(uploaded_files) > 1:
            # Si plusieurs images ont été uploadées
            st.write("Analyse de l'ensemble des fichiers importés.")
            
            # Bouton pour lancer le traitement batch
            if st.button(f"LANCER LE TRAITEMENT SUR {len(uploaded_files)} FICHIERS"):
                
                if status:
                    # Si le modèle est chargé
                    
                    all_detections = []  # Liste qui stockera toutes les détections
                    
                    # Création d'une barre de progression
                    progress_bar = st.progress(0)
                    
                    # ========== BOUCLE SUR TOUTES LES IMAGES ==========
                    for i, file in enumerate(uploaded_files):
                        # Réinitialisation du curseur
                        file.seek(0)
                        
                        # Lecture et décodage de l'image
                        f_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        im = cv2.imdecode(f_bytes, 1)
                        
                        # Prétraitement
                        im_proc = process_signal(im, bri, con, gam, sha)
                        
                        # Mesure du temps de traitement
                        t0 = time.perf_counter()
                        
                        # Inférence
                        dets = run_inference_pytorch(model, im_proc, conf_thresh, nms_thresh)
                        
                        # Calcul du temps en ms
                        dt = (time.perf_counter() - t0) * 1000
                        
                        # ========== AJOUT DES RÉSULTATS ==========
                        if dets:
                            # Si des avions ont été détectés
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
                            # Si aucune détection, ajoute une ligne "R.A.S" (Rien À Signaler)
                            all_detections.append({
                                "Image": file.name,
                                "Classe": "R.A.S",
                                "Précision": 0.0,
                                "Temps (ms)": int(dt),
                                "Coordonnées": "-"
                            })
                        
                        # Mise à jour de la barre de progression
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # ========== AFFICHAGE DES RÉSULTATS BATCH ==========
                    if all_detections:
                        # Conversion en DataFrame
                        df_all = pd.DataFrame(all_detections)
                        
                        # Tri par précision décroissante et formatage
                        df_display = df_all.sort_values(by="Précision", ascending=False).copy()
                        df_display["Précision"] = df_display["Précision"].apply(lambda x: f"{x:.2%}")
                        
                        # Message de succès
                        st.success(f"Scan terminé : {len(all_detections)} objets traités.")
                        
                        # Affichage du tableau complet
                        st.dataframe(df_display, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # ========== GRAPHIQUE DE RÉPARTITION DES CLASSES ==========
                        st.subheader("📊 Répartition des Classes")
                        
                        # Filtrage pour exclure "R.A.S"
                        df_chart = df_all[df_all['Classe'] != "R.A.S"]
                        
                        if not df_chart.empty:
                            # Graphique en barres comptant les occurrences de chaque classe
                            # value_counts() compte le nombre de fois que chaque classe apparaît
                            st.bar_chart(df_chart['Classe'].value_counts())
                        else:
                            st.info("Rien à signaler.")
                    else:
                        st.warning("Aucune détection.")
                else:
                    # Si le modèle n'est pas chargé
                    st.error("Modèle non chargé.")
        else:
            # Si une seule image a été uploadée
            st.info("Chargez plusieurs images pour activer le Scan Global.")
else:
    # Si aucune image n'a été uploadée
    st.info("En attente d'images satellite...")
