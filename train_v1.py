"""
train_v1.py - Entraînement Baseline (Version 1)

Ce script effectue le premier entraînement du modèle Faster R-CNN.
Il part des poids pré-entraînés sur COCO et les adapte à notre dataset d'avions.
C'est la version "Baseline" qui sert de référence pour mesurer les améliorations.

Stratégie:
- Split 80/20 avec réserve de 50 images pour test final
- Learning Rate standard (0.005) pour un premier entraînement
- 10 époques pour permettre au modèle de bien converger
"""

import torch
import os
import config  # Import de la configuration centralisée (chemins, hyperparamètres)
from model import get_model_instance_segmentation  # Architecture Faster R-CNN
from dataset import PlaneDataset  # Classe de chargement des données
import torchvision.transforms as T  # Transformations d'images
from torch.utils.data import DataLoader, Subset  # Outils de chargement par batch

def get_transform():
    """
    Définit les transformations à appliquer aux images.
    
    Returns:
        T.Compose: Pipeline de transformations
        - ToTensor(): Convertit l'image PIL (0-255) en tensor PyTorch (0-1)
          et change le format de (H, W, C) à (C, H, W)
    """
    return T.Compose([T.ToTensor()])

# --- CONFIGURATION VIA CONFIG.PY ---
# Récupération du device (GPU si disponible, sinon CPU)
DEVICE = config.DEVICE

# ========== 1. PRÉPARATION DES DONNÉES ==========
# Chargement du dataset complet depuis le dossier data/
# PlaneDataset lit les images et annotations XML (format PASCAL VOC)
dataset = PlaneDataset(config.DATA_DIR, transforms=get_transform())

# Création d'un split train/test aléatoire
# torch.randperm() génère une permutation aléatoire des indices [0, 1, 2, ..., N-1]
indices = torch.randperm(len(dataset)).tolist()

# On garde toutes les images SAUF les 50 dernières (après mélange)
# Ces 50 images serviront de test final pour évaluer le modèle V1
# Exemple: Si 1331 images → dataset_train contient 1281 images
dataset_train = Subset(dataset, indices[:-50])

# DataLoader: charge les données par batchs (groupes d'images)
data_loader = DataLoader(
    dataset_train,  # Sous-dataset d'entraînement
    batch_size=4,   # Traite 4 images à la fois (limité par mémoire GPU)
    shuffle=True,   # Mélange les images à chaque époque (évite apprentissage de l'ordre)
    collate_fn=lambda x: tuple(zip(*x))  # Fonction pour assembler les batchs
    # collate_fn nécessaire car les images ont des tailles différentes
    # Transforme [(img1, target1), (img2, target2)] en ([img1, img2], [target1, target2])
)

# ========== 2. CRÉATION DU MODÈLE ==========
# Initialisation du modèle Faster R-CNN avec 21 classes (20 avions + 1 background)
model = get_model_instance_segmentation(config.NUM_CLASSES)

# Déplacement du modèle sur le GPU (ou CPU si GPU indisponible)
# Cette étape est CRUCIALE pour utiliser le GPU
model.to(DEVICE)

# ========== 3. CONFIGURATION DE L'OPTIMISEUR ==========
# Récupération des paramètres entraînables du modèle
# requires_grad=True signifie que le paramètre sera mis à jour pendant l'entraînement
params = [p for p in model.parameters() if p.requires_grad]

# Optimiseur SGD (Stochastic Gradient Descent)
# Algorithme qui ajuste les poids du modèle pour minimiser la loss
optimizer = torch.optim.SGD(
    params,              # Paramètres à optimiser
    lr=0.005,            # Learning Rate: taille du pas de descente (standard pour V1)
    momentum=0.9,        # Momentum: accélère la convergence en mémorisant la direction
    weight_decay=0.0005  # Régularisation L2: pénalise les poids trop élevés (évite overfitting)
)

# ========== 4. BOUCLE D'ENTRAÎNEMENT ==========
num_epochs = 10  # Nombre de passages complets sur le dataset
print(f"🚀 Début de l'entraînement Baseline (V1) sur : {DEVICE}")

# Boucle principale: répète l'entraînement sur toutes les données 10 fois
for epoch in range(num_epochs):
    
    # Mode entraînement: active dropout, batch normalization, etc.
    model.train()
    
    # Compteur d'itérations pour affichage
    i = 0
    
    # Itération sur les batchs de données
    for images, targets in data_loader:
        # Déplacement des images sur le GPU/CPU
        # List comprehension qui applique .to(DEVICE) à chaque image
        images = list(image.to(DEVICE) for image in images)
        
        # Déplacement des annotations (targets) sur le GPU/CPU
        # Dict comprehension imbriquée: pour chaque target, déplace toutes ses valeurs
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Forward pass: le modèle calcule la loss automatiquement en mode train
        # Faster R-CNN calcule 4 losses (RPN classification, RPN regression, ROI classification, ROI regression)
        loss_dict = model(images, targets)
        
        # Somme des 4 losses pour obtenir la loss totale
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass: calcul des gradients
        # Étape 1: Réinitialiser les gradients à zéro (PyTorch les accumule par défaut)
        optimizer.zero_grad()
        
        # Étape 2: Rétropropagation - calcule les gradients de la loss par rapport à chaque poids
        losses.backward()
        
        # Étape 3: Mise à jour des poids selon la formule SGD avec momentum
        optimizer.step()

        # Affichage de la loss tous les 10 batchs pour suivre la progression
        if i % 10 == 0:
            # .item() convertit le tensor en float Python pour l'affichage
            print(f"Époque {epoch+1}, Itération {i}, Loss: {losses.item():.4f}")
        i += 1

# ========== 5. SAUVEGARDE DU MODÈLE ==========
# Construction du chemin de sauvegarde dans le dossier models/
save_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions.pth')

# Sauvegarde des poids du modèle (state_dict = dictionnaire de tous les paramètres)
# Seuls les poids sont sauvegardés, pas l'architecture ni l'optimiseur
torch.save(model.state_dict(), save_path)

print(f"✅ Entraînement V1 terminé ! Modèle sauvegardé dans : {save_path}")
