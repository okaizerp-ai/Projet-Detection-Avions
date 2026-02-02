"""
train_v2.py - Fine-Tuning (Version 2)

Ce script effectue le fine-tuning du modèle V1 pour améliorer ses performances.
Il charge les poids de la V1 et continue l'entraînement avec:
- Un learning rate plus faible (0.0005 vs 0.005) pour ajustements précis
- Tout le dataset (pas de split) pour maximiser les données d'apprentissage
- Data augmentation ciblée sur les classes faibles (dans dataset.py)
- Barre de progression (tqdm) pour suivi visuel

Objectif: Gagner 5 points de F1-Score par rapport à V1 (69% → 74%)
"""

import torch
import os
import config  # Configuration centralisée (chemins, device, hyperparamètres)
from model import get_model_instance_segmentation  # Architecture Faster R-CNN
from dataset import PlaneDataset  # Dataset avec data augmentation intégrée
import torchvision.transforms as T  # Transformations d'images
from tqdm import tqdm  # Bibliothèque pour barres de progression visuelles

# --- CONFIGURATION VIA CONFIG.PY ---
# Récupération automatique du device (cuda ou cpu)
DEVICE = config.DEVICE

# ========== 1. CHARGEMENT DU MODÈLE V1 (TRANSFER LEARNING INTERNE) ==========
# Création de l'architecture avec 21 classes (identique à V1)
model = get_model_instance_segmentation(config.NUM_CLASSES)

# Construction du chemin vers le modèle V1 sauvegardé
checkpoint_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions.pth')

# Vérification de l'existence du fichier de poids V1
if os.path.exists(checkpoint_path):
    print(f"💎 Chargement du modèle existant (V1) depuis : {checkpoint_path}")
    
    # Chargement des poids V1 dans le modèle
    # torch.load() désérialise le fichier .pth en dictionnaire de tensors
    # map_location=DEVICE assure la compatibilité GPU/CPU (charge sur le bon device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
else:
    # Si V1 n'existe pas, on part des poids COCO pré-entraînés
    print(f"⚠️ Attention : {checkpoint_path} non trouvé. Le Fine-Tuning partira de zéro.")

# Déplacement du modèle sur GPU/CPU
model.to(DEVICE)

# ========== 2. OPTIMISEUR AVEC LEARNING RATE FAIBLE ==========
# Récupération des paramètres entraînables
params = [p for p in model.parameters() if p.requires_grad]

# Optimiseur SGD avec LR divisé par 10 par rapport à V1
optimizer = torch.optim.SGD(
    params,
    lr=0.0005,           # Learning Rate FAIBLE (0.0005 vs 0.005 en V1)
                         # Raison: Fine-tuning nécessite des ajustements délicats
                         # Un LR trop élevé "casserait" les connaissances de V1
    momentum=0.9,        # Même momentum qu'en V1 (valeur standard)
    weight_decay=0.0005  # Même régularisation L2 qu'en V1
)

# ========== 3. DATALOADER AVEC TOUT LE DATASET ==========
# Chargement du dataset COMPLET (toutes les ~1331 images)
# PlaneDataset applique automatiquement la data augmentation ciblée:
# - 80% de flip pour classes faibles (A1, A12, A15, A18, A20)
# - 50% de flip pour les autres classes
dataset = PlaneDataset(config.DATA_DIR, transforms=T.Compose([T.ToTensor()]))

# DataLoader pour charger les données par batchs
data_loader = torch.utils.data.DataLoader(
    dataset,             # Dataset complet (pas de Subset comme en V1)
    batch_size=4,        # Même batch size qu'en V1
    shuffle=True,        # Mélange à chaque époque
    num_workers=0,       # Pas de processus parallèles (plus stable sur Windows/Mac)
                         # num_workers=4 serait plus rapide mais peut causer des bugs
    collate_fn=lambda x: tuple(zip(*x))  # Assemblage des batchs (voir train_v1.py)
)

# ========== 4. BOUCLE D'ENTRAÎNEMENT AVEC TQDM ==========
num_epochs = 7  # Moins d'époques qu'en V1 (7 vs 10) car on part de V1 déjà entraîné
print(f"🚀 Début du Fine-Tuning V2 sur : {DEVICE}")

# Boucle principale sur les époques
for epoch in range(num_epochs):
    
    # Mode entraînement (active dropout, batch norm, etc.)
    model.train()
    
    # Création de la barre de progression pour l'époque actuelle
    # tqdm() enveloppe le DataLoader et affiche [████░░] 67% | loss=0.234
    prog_bar = tqdm(data_loader, desc=f"Époque {epoch+1}/{num_epochs}")
    
    # Accumulation de la loss pour calcul de moyenne en fin d'époque
    epoch_loss = 0
    
    # Itération sur les batchs avec barre de progression
    for images, targets in prog_bar:
        # Déplacement des données sur GPU/CPU (identique à V1)
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # Forward pass: calcul de la loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass: calcul des gradients et mise à jour des poids
        optimizer.zero_grad()  # Reset gradients
        losses.backward()       # Calcul gradients
        optimizer.step()        # Mise à jour poids
        
        # Accumulation de la loss pour statistiques
        epoch_loss += losses.item()
        
        # Mise à jour de la barre de progression avec la loss actuelle
        # Affiche la loss à droite de la barre: [████░░] | loss=0.234
        prog_bar.set_postfix(loss=losses.item())
    
    # Calcul de la loss moyenne de l'époque
    # len(data_loader) = nombre de batchs dans l'époque
    avg_loss = epoch_loss / len(data_loader)
    
    # Affichage de la loss moyenne (indicateur de progression)
    # Cette valeur doit diminuer au fil des époques
    print(f"✅ Époque {epoch+1} terminée. Perte moyenne : {avg_loss:.4f}")

# ========== 5. SAUVEGARDE DU MODÈLE V2 ==========
# Sauvegarde sous un nom différent pour garder V1 intact
save_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')

# Sauvegarde des poids uniquement (pas l'architecture ni l'optimiseur)
torch.save(model.state_dict(), save_path)

print(f"✨ Bravo ! Le modèle V2 est sauvegardé ici : {save_path}")
