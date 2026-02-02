"""
confusion.py - Génération de la Matrice de Confusion

Ce script génère une matrice de confusion pour évaluer les performances du modèle V2.
La matrice de confusion est un tableau 20x20 qui montre pour chaque classe réelle (lignes)
combien de fois elle a été prédite dans chaque classe (colonnes).

Utilité:
- Diagonale = prédictions correctes (plus la diagonale est foncée, mieux c'est)
- Hors diagonale = confusions entre classes (révèle quels avions sont confondus)
- Permet d'identifier les faiblesses du modèle (ex: A15 confondu avec A13)

La matrice est normalisée en pourcentages pour faciliter la lecture.
"""

import torch  # PyTorch pour charger le modèle et faire l'inférence
import numpy as np  # NumPy pour calculs matriciels (normalisation, etc.)
import matplotlib.pyplot as plt  # Matplotlib pour créer le graphique
import seaborn as sns  # Seaborn pour améliorer le rendu visuel de la heatmap
from sklearn.metrics import confusion_matrix  # Fonction de scikit-learn pour calculer la matrice
import os  # Manipulation des chemins
from tqdm import tqdm  # Barre de progression pour suivre l'avancement
import torchvision.transforms as T  # Transformations d'images

# Import des fichiers locaux du projet
import config  # Configuration centralisée (chemins, device, classes)
from model import get_model_instance_segmentation  # Architecture Faster R-CNN
from dataset import PlaneDataset  # Dataset personnalisé

def run_full_confusion_matrix():
    """
    Fonction principale qui orchestre la génération de la matrice de confusion.
    
    Étapes:
    1. Charge le modèle V2 entraîné
    2. Effectue l'inférence sur toutes les images du dataset
    3. Compare les prédictions aux annotations réelles (ground truth)
    4. Calcule et normalise la matrice de confusion
    5. Génère et sauvegarde le graphique
    """
    
    # ========== AFFICHAGE DES INFORMATIONS SYSTÈME ==========
    print(f"⚙️ Racine du projet : {config.ROOT_DIR}")
    print(f"🖥️ Calcul de la matrice sur : {config.DEVICE}")

    # ========== 1. CONSTRUCTION DES CHEMINS ==========
    # Chemin vers le modèle V2 entraîné (dans le dossier models/)
    model_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
    
    # Chemin de sortie pour sauvegarder l'image de la matrice (dans outputs/)
    output_path = os.path.join(config.OUTPUTS_DIR, 'matrice_confusion_TOTALE.png')

    # ========== 2. CHARGEMENT DU MODÈLE V2 ==========
    # Création de l'architecture (21 classes)
    model = get_model_instance_segmentation(config.NUM_CLASSES)
    
    # Vérification de l'existence du fichier .pth
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable à {model_path}")
        return  # Sort de la fonction si le modèle n'existe pas

    # Chargement des poids V2 dans le modèle
    # map_location=config.DEVICE assure la compatibilité GPU/CPU
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    
    # Déplacement du modèle sur GPU/CPU et activation du mode évaluation
    model.to(config.DEVICE)
    model.eval()  # Mode eval: désactive dropout, fige batch norm (crucial pour inférence)

    # ========== 3. PRÉPARATION DU DATASET COMPLET ==========
    # Chargement de TOUTES les images du dataset (environ 1331 images)
    # ToTensor() convertit les images PIL en tensors PyTorch
    dataset = PlaneDataset(config.DATA_DIR, transforms=T.Compose([T.ToTensor()]))
    
    # DataLoader pour charger les images une par une
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Une image à la fois (simplifie la comparaison prédiction vs vérité)
        shuffle=False,  # Pas de mélange (pas nécessaire pour l'évaluation)
        collate_fn=lambda x: tuple(zip(*x))  # Assemblage des batchs
    )

    # ========== 4. INITIALISATION DES LISTES DE COLLECTE ==========
    # Liste qui stockera toutes les prédictions du modèle
    all_preds = []
    
    # Liste qui stockera toutes les vraies classes (ground truth des annotations XML)
    all_gt = []

    print(f"🔎 Analyse complète de {len(dataset)} images en cours...")
    
    # ========== 5. BOUCLE D'INFÉRENCE ==========
    # torch.no_grad() désactive le calcul des gradients (économise mémoire et accélère)
    with torch.no_grad():
        
        # Itération sur toutes les images avec barre de progression
        for images, targets in tqdm(data_loader):
            
            # Déplacement des images sur GPU/CPU
            # List comprehension qui applique .to(DEVICE) à chaque image
            images = [img.to(config.DEVICE) for img in images]
            
            # Inférence: le modèle retourne les prédictions
            # outputs est une liste de dicts [{boxes, labels, scores}, ...]
            outputs = model(images)
            
            # ========== TRAITEMENT DE CHAQUE IMAGE DU BATCH ==========
            # (ici batch_size=1 donc une seule itération)
            for i in range(len(targets)):
                
                # Extraction des labels réels (ground truth) depuis les annotations
                # .cpu() déplace le tensor du GPU vers le CPU
                # .numpy() convertit le tensor PyTorch en array NumPy
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                # Extraction des labels prédits par le modèle
                pred_labels = outputs[i]['labels'].cpu().numpy()
                
                # Extraction des scores de confiance des prédictions
                pred_scores = outputs[i]['scores'].cpu().numpy()
                
                # ========== FILTRAGE DES PRÉDICTIONS PAR SEUIL DE CONFIANCE ==========
                # Seuil à 0.4 (40%) pour capter même les détections incertaines
                # Plus bas que 0.5 pour ne pas rater les vraies détections avec confiance moyenne
                mask = pred_scores > 0.4
                
                # Indexation booléenne: garde seulement les labels où mask=True
                # valid_preds contient les labels des détections > 40% de confiance
                valid_preds = pred_labels[mask]
                
                # ========== AJOUT À LA LISTE SI DÉTECTION VALIDE ==========
                # Vérifie qu'il y a au moins une prédiction valide ET au moins un objet réel
                if len(valid_preds) > 0 and len(gt_labels) > 0:
                    # SIMPLIFICATION: On compare seulement le 1er avion prédit au 1er avion réel
                    # Une vraie matrice devrait matcher les boîtes par IoU (Intersection over Union)
                    # mais cette simplification suffit pour avoir une vue d'ensemble
                    all_preds.append(valid_preds[0])  # Premier avion prédit
                    all_gt.append(gt_labels[0])       # Premier avion réel

    # ========== 6. CALCUL DE LA MATRICE DE CONFUSION ==========
    # labels_range : Liste des classes [1, 2, ..., 20] (on ignore Background=0)
    labels_range = list(range(1, 21))
    
    # confusion_matrix() de scikit-learn calcule la matrice
    # cm[i, j] = nombre de fois où la vraie classe i a été prédite comme j
    # Retourne une matrice NumPy 20x20
    cm = confusion_matrix(all_gt, all_preds, labels=labels_range)

    # ========== 7. NORMALISATION DE LA MATRICE (CONVERSION EN POURCENTAGES) ==========
    # Contexte pour ignorer les warnings de division par zéro (si une classe n'a aucun exemple)
    with np.errstate(divide='ignore', invalid='ignore'):
        
        # Conversion en float pour permettre la division
        # cm.sum(axis=1) : Somme par ligne (total de chaque classe réelle)
        # [:, np.newaxis] : Transforme array 1D en colonne 2D pour broadcasting
        # Division : chaque ligne est divisée par son total
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # np.nan_to_num() remplace les NaN (0/0) par 0
        # Se produit si une classe n'apparaît jamais dans le dataset
        cm_norm = np.nan_to_num(cm_norm)

    # ========== 8. CRÉATION DU GRAPHIQUE ==========
    # Création d'une figure de grande taille (18x14 pouces) pour accueillir 20x20 cases
    plt.figure(figsize=(18, 14))
    
    # sns.heatmap() crée une carte de chaleur (heatmap) colorée
    sns.heatmap(
        cm_norm,           # Données: matrice normalisée
        annot=True,        # Affiche les valeurs numériques dans chaque case
        fmt='.2f',         # Format à 2 décimales (ex: 0.98)
        cmap='Greens',     # Palette de couleurs (blanc → vert foncé)
        xticklabels=config.CLASSES[1:],  # Labels des colonnes (A1 à A20)
        yticklabels=config.CLASSES[1:]   # Labels des lignes (A1 à A20)
    )
    # config.CLASSES[1:] : Slice qui prend tout sauf Background (index 0)
    
    # ========== 9. AJOUT DES LABELS ET TITRE ==========
    plt.title(f'Matrice de Confusion Globale ({len(dataset)} images) - Modèle Elite', fontsize=18)
    plt.xlabel('Prédictions de l\'IA', fontsize=14)  # Axe X = ce que le modèle prédit
    plt.ylabel('Vérité Terrain (Annotations XML)', fontsize=14)  # Axe Y = vraie classe
    
    # ========== 10. SAUVEGARDE ET AFFICHAGE ==========
    # Sauvegarde de la figure en PNG dans le dossier outputs/
    plt.savefig(output_path)
    
    # Affichage de la figure à l'écran (fenêtre interactive)
    plt.show()
    
    print(f"✅ Matrice finale sauvegardée dans : {output_path}")

# ========== POINT D'ENTRÉE DU SCRIPT ==========
# Exécute run_full_confusion_matrix() si le script est lancé directement
# (pas si importé avec 'import confusion')
if __name__ == "__main__":
    run_full_confusion_matrix()
