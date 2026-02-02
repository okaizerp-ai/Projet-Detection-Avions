"""
generate_json.py - Génération du JSON d'évaluation

Ce script effectue l'inférence (prédiction) du modèle V2 sur les images d'évaluation
fournies par le professeur et exporte les résultats au format JSON.

Le JSON généré contient pour chaque image:
- La classe prédite (A1 à A20)
- Le score de confiance (0 à 1)
- Les coordonnées de la bounding box (xmin, ymin, xmax, ymax)

Ce fichier sera utilisé par le professeur pour évaluer automatiquement le modèle.

Format du JSON de sortie:
{
  "image1.jpg": [
    {
      "class": "A5",
      "score": 0.98,
      "coordinates": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400}
    },
    ...
  ],
  "image2.jpg": [...]
}
"""

import torch
import torchvision.transforms as T  # Transformations d'images
from PIL import Image  # Bibliothèque de manipulation d'images
import os
import json  # Bibliothèque pour lire/écrire des fichiers JSON
import config  # Configuration centralisée (chemins, device, classes)
from model import get_model_instance_segmentation  # Architecture Faster R-CNN
from tqdm import tqdm  # Barre de progression pour suivi visuel

def run_evaluation():
    """
    Fonction principale qui orchestre l'évaluation complète:
    1. Charge le modèle V2 entraîné
    2. Lit toutes les images d'évaluation
    3. Effectue l'inférence (prédiction) sur chaque image
    4. Exporte les résultats au format JSON
    """
    
    print(f"🚀 Inférence universelle sur : {config.DEVICE}")
    
    # ========== 1. CHARGEMENT DU MODÈLE V2 ==========
    # Création de l'architecture (21 classes)
    model = get_model_instance_segmentation(config.NUM_CLASSES)
    
    # Construction du chemin vers le modèle V2 sauvegardé
    model_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
    
    # Vérification de l'existence du fichier
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable dans : {model_path}")
        return  # Sort de la fonction si le modèle n'existe pas
    
    # Chargement des poids V2 dans le modèle
    # map_location=DEVICE gère la compatibilité GPU/CPU
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    
    # Déplacement du modèle sur GPU/CPU et activation du mode évaluation
    model.to(config.DEVICE)
    model.eval()  # Mode eval: désactive dropout, fige batch norm (crucial pour inférence)

    # ========== 2. LISTE DES IMAGES D'ÉVALUATION ==========
    # Récupération du dossier d'évaluation depuis config
    eval_dir = config.EVAL_IMG_DIR
    
    # Vérification de l'existence du dossier
    if not os.path.exists(eval_dir):
        print(f"❌ Dossier images introuvable : {eval_dir}")
        return  # Sort si le dossier n'existe pas
    
    # Liste de tous les fichiers image du dossier
    # List comprehension avec filtre sur les extensions image
    # .lower() pour gérer .JPG, .JPEG, .PNG (majuscules)
    all_images = [f for f in os.listdir(eval_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Dictionnaire qui contiendra tous les résultats
    # Structure: {nom_image: [liste_détections]}
    results = {}

    print(f"🔎 Analyse de {len(all_images)} images pour le JSON...")

    # ========== 3. BOUCLE DE DÉTECTION ==========
    # torch.no_grad() désactive le calcul des gradients
    # Économise mémoire et accélère les calculs (pas de backprop en inférence)
    with torch.no_grad():
        
        # Itération sur toutes les images avec barre de progression
        for img_name in tqdm(all_images):
            
            # Construction du chemin complet vers l'image
            img_path = os.path.join(eval_dir, img_name)
            
            # Ouverture de l'image avec PIL et conversion en RGB
            # .convert("RGB") force 3 canaux même si l'image est en niveaux de gris
            img = Image.open(img_path).convert("RGB")
            
            # Préparation de l'image pour le modèle:
            # 1. T.ToTensor(): PIL (H,W,C) [0-255] → Tensor (C,H,W) [0-1]
            # 2. .unsqueeze(0): Ajoute dimension batch (C,H,W) → (1,C,H,W)
            #    Le modèle attend toujours un batch, même d'une seule image
            # 3. .to(DEVICE): Déplace le tensor sur GPU/CPU
            img_t = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(config.DEVICE)
            
            # Inférence: le modèle retourne les prédictions
            # prediction est une liste de 1 dict (car batch_size=1):
            # [{
            #     'boxes': tensor([[x1,y1,x2,y2], ...]),   # Coordonnées des boîtes
            #     'labels': tensor([5, 12, 3]),             # Classes prédites
            #     'scores': tensor([0.98, 0.87, 0.65])      # Confiances
            # }]
            prediction = model(img_t)
            
            # Liste qui contiendra les détections de cette image
            img_preds = []
            
            # Extraction des tensors de prédiction et conversion en NumPy
            # .cpu() déplace les tensors du GPU vers le CPU (PIL ne comprend pas CUDA)
            # .numpy() convertit les tensors PyTorch en arrays NumPy (format plus standard)
            boxes = prediction[0]['boxes'].cpu().numpy()    # Array shape (N, 4)
            labels = prediction[0]['labels'].cpu().numpy()  # Array shape (N,)
            scores = prediction[0]['scores'].cpu().numpy()  # Array shape (N,)
            
            # Boucle sur toutes les détections de cette image
            for i in range(len(boxes)):
                
                # Filtrage par seuil de confiance: garde seulement les détections > 50%
                # Élimine les faux positifs (détections incertaines)
                if scores[i] > 0.5:
                    
                    # Extraction de la boîte actuelle (array NumPy [xmin, ymin, xmax, ymax])
                    box = boxes[i]
                    
                    # Construction du dictionnaire de détection au format JSON
                    img_preds.append({
                        # Conversion de l'index (ex: 5) en label (ex: "A5")
                        "class": config.CLASSES[labels[i]],
                        
                        # Conversion du tensor en float Python pour sérialisation JSON
                        "score": float(scores[i]),
                        
                        # Dictionnaire des coordonnées (conversion en int = pixels entiers)
                        "coordinates": {
                            "xmin": int(box[0]),  # Coin supérieur gauche X
                            "ymin": int(box[1]),  # Coin supérieur gauche Y
                            "xmax": int(box[2]),  # Coin inférieur droit X
                            "ymax": int(box[3])   # Coin inférieur droit Y
                        }
                    })
            
            # Association du nom d'image à sa liste de détections dans le dict global
            results[img_name] = img_preds

    # ========== 4. SAUVEGARDE DU JSON ==========
    # Construction du chemin de sortie dans le dossier outputs/
    output_json = os.path.join(config.OUTPUTS_DIR, 'predictions_officielles.json')
    
    # Ouverture du fichier en mode écriture ('w' = write)
    # 'with' assure la fermeture automatique du fichier même en cas d'erreur
    with open(output_json, 'w') as f:
        # Sérialisation du dictionnaire Python en JSON
        # indent=2 formate le JSON avec indentation (lisible par humain)
        # Sans indent, tout serait sur une seule ligne
        json.dump(results, f, indent=2)

    print(f"\n✅ JSON généré avec succès dans : {output_json}")

# ========== POINT D'ENTRÉE DU SCRIPT ==========
# Exécute run_evaluation() seulement si le script est lancé directement
# (pas si importé avec 'import generate_json')
if __name__ == "__main__":
    run_evaluation()
