"""
detection.py - Génération d'Images Annotées avec Détections Visuelles

Ce script effectue l'inférence du modèle V2 sur les images d'évaluation et génère
des visualisations avec les boîtes englobantes et labels dessinés directement sur les images.

Utilité:
- Permet de vérifier visuellement les performances du modèle
- Utile pour la présentation et le rapport (captures d'écran des détections)
- Aide à identifier les erreurs du modèle (faux positifs, faux négatifs)

Les images annotées sont sauvegardées dans outputs/DETECTIONS_VISUELLES/
avec le préfixe "visu_" ajouté au nom original.
"""

import torch  # PyTorch pour charger le modèle et faire l'inférence
import torchvision.transforms as T  # Transformations d'images
from PIL import Image, ImageDraw, ImageFont  # Bibliothèque PIL pour dessiner sur les images
import os  # Manipulation des chemins
import config  # Configuration centralisée (chemins, device, classes)
from model import get_model_instance_segmentation  # Architecture Faster R-CNN
from tqdm import tqdm  # Barre de progression pour suivre l'avancement

# ========== CONFIGURATION DES CHEMINS ==========
# Chemin vers le modèle V2 entraîné (dans le dossier models/)
MODEL_PATH = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')

# Dossier de sortie pour les images annotées (dans outputs/)
OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, 'DETECTIONS_VISUELLES')

# Création du dossier de sortie s'il n'existe pas
# exist_ok=True évite une erreur si le dossier existe déjà
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. CHARGEMENT DU MODÈLE V2 ==========
# Création de l'architecture Faster R-CNN avec 21 classes
model = get_model_instance_segmentation(config.NUM_CLASSES)

# Chargement des poids du modèle V2 depuis le fichier .pth
# map_location=config.DEVICE assure la compatibilité GPU/CPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))

# Déplacement du modèle sur GPU/CPU et activation du mode évaluation
model.to(config.DEVICE)
model.eval()  # Mode eval: désactive dropout, fige batch norm (crucial pour inférence)

# ========== 2. LISTE DES IMAGES À TRAITER ==========
# Récupération du dossier contenant les images d'évaluation du professeur
img_dir = config.EVAL_IMG_DIR

# Liste de tous les fichiers image du dossier
# List comprehension avec filtre sur les extensions d'images
# .lower() gère les extensions en majuscules (.JPG, .JPEG, .PNG)
# .endswith(tuple) vérifie si le nom se termine par une des extensions
all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# ========== 3. CHARGEMENT DE LA POLICE DE CARACTÈRES ==========
# Tentative de charger une police TrueType système (Linux)
try:
    # ImageFont.truetype() charge une police avec taille spécifiée (ici 20 points)
    # Chemin typique sur Ubuntu/Debian
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
except:
    # Si la police n'existe pas (Windows, Mac, autre Linux), utilise la police par défaut
    # La police par défaut est basique mais fonctionne partout
    font = ImageFont.load_default()

# Message d'information sur le début du traitement
print(f"🎨 Génération des visuels dans : {OUTPUT_DIR}")

# ========== 4. BOUCLE DE DÉTECTION ET ANNOTATION ==========
# torch.no_grad() désactive le calcul des gradients (économise mémoire et accélère)
with torch.no_grad():
    
    # Itération sur toutes les images avec barre de progression
    for img_name in tqdm(all_images):
        
        # ========== CHARGEMENT DE L'IMAGE ==========
        # Construction du chemin complet vers l'image
        img_path = os.path.join(img_dir, img_name)
        
        # Ouverture de l'image avec PIL et conversion en RGB
        # .convert("RGB") force 3 canaux même si l'image est en niveaux de gris
        img = Image.open(img_path).convert("RGB")
        
        # ========== PRÉPARATION DE L'IMAGE POUR LE MODÈLE ==========
        # ToTensor() convertit PIL (H,W,C) [0-255] → Tensor (C,H,W) [0-1]
        # .unsqueeze(0) ajoute la dimension batch: (C,H,W) → (1,C,H,W)
        # .to(DEVICE) déplace le tensor sur GPU/CPU
        img_tensor = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(config.DEVICE)
        
        # ========== INFÉRENCE ==========
        # Le modèle retourne une liste de 1 dict (car batch_size=1):
        # [{'boxes': tensor, 'labels': tensor, 'scores': tensor}]
        prediction = model(img_tensor)
        
        # ========== CRÉATION DE L'OBJET DE DESSIN ==========
        # ImageDraw.Draw() crée un contexte de dessin lié à l'image PIL
        # Permet de dessiner des rectangles, du texte, etc. directement sur l'image
        draw = ImageDraw.Draw(img)
        
        # ========== EXTRACTION DES RÉSULTATS ==========
        # Déplacement des tensors du GPU vers CPU et conversion en NumPy
        # prediction[0] car c'est une liste d'un seul élément (batch_size=1)
        boxes = prediction[0]['boxes'].cpu().numpy()    # Array shape (N, 4)
        scores = prediction[0]['scores'].cpu().numpy()  # Array shape (N,)
        labels = prediction[0]['labels'].cpu().numpy()  # Array shape (N,)

        # ========== DESSIN DES DÉTECTIONS ==========
        # Boucle sur toutes les détections (N objets détectés)
        for i in range(len(boxes)):
            
            # ========== FILTRAGE PAR SEUIL DE CONFIANCE ==========
            # Garde seulement les détections avec score > 50%
            # Élimine les faux positifs (détections incertaines)
            if scores[i] > 0.5:
                
                # Extraction des coordonnées de la boîte (array NumPy [xmin, ymin, xmax, ymax])
                box = boxes[i]
                
                # ========== CONSTRUCTION DU LABEL TEXTE ==========
                # Convertit l'index (ex: 5) en label (ex: "A5")
                # Ajoute le score en pourcentage (ex: "(98%)")
                # int(scores[i]*100) convertit 0.987 → 98
                label_txt = f"{config.CLASSES[labels[i]]} ({int(scores[i]*100)}%)"
                
                # ========== DESSIN DE LA BOÎTE ENGLOBANTE ==========
                # draw.rectangle() dessine un rectangle
                # Coordonnées: [(coin_supérieur_gauche), (coin_inférieur_droit)]
                # outline="lime" : Couleur du contour (vert fluo)
                # width=4 : Épaisseur du trait en pixels
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="lime", width=4)
                
                # ========== DESSIN DU FOND DU LABEL ==========
                # Rectangle de fond pour rendre le texte lisible
                # Position: juste au-dessus de la boîte principale
                # [(xmin, ymin-25), (xmin+130, ymin)] : rectangle de 130x25 pixels
                # fill="lime" : Couleur de remplissage (vert fluo)
                draw.rectangle([(box[0], box[1] - 25), (box[0] + 130, box[1])], fill="lime")
                
                # ========== DESSIN DU TEXTE ==========
                # draw.text() dessine du texte
                # Position: (xmin+5, ymin-22) = 5 pixels à droite, 22 pixels au-dessus
                # Légèrement décalé pour ne pas toucher les bords du rectangle vert
                # fill="black" : Couleur du texte (noir sur fond vert = bon contraste)
                # font=font : Utilise la police chargée précédemment
                draw.text((box[0] + 5, box[1] - 22), label_txt, fill="black", font=font)

        # ========== SAUVEGARDE DE L'IMAGE ANNOTÉE ==========
        # .save() sauvegarde l'image PIL modifiée
        # Nom de sortie: "visu_" + nom original (ex: "visu_image_45.jpg")
        # Sauvegardée dans le dossier OUTPUT_DIR
        img.save(os.path.join(OUTPUT_DIR, f"visu_{img_name}"))

# Message de confirmation à la fin du traitement
print(f"\n✅ Terminé !")
