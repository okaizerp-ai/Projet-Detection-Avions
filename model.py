"""
model.py - Définition de l'architecture Faster R-CNN

Ce fichier crée le modèle de détection d'objets utilisé pour le projet.
Il utilise le Transfer Learning en chargeant un Faster R-CNN pré-entraîné
sur le dataset COCO, puis adapte la couche de classification pour nos 21 classes.
"""

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    """
    Crée et configure un modèle Faster R-CNN pour la détection d'avions.
    
    Args:
        num_classes (int): Nombre total de classes (20 avions + 1 background = 21)
    
    Returns:
        model: Modèle Faster R-CNN configuré et prêt à être entraîné
    
    Architecture:
        - Backbone: ResNet50 avec Feature Pyramid Network (FPN)
        - RPN: Region Proposal Network pour proposer les zones d'objets
        - ROI Head: Classification et régression des boîtes englobantes
    """
    
    # Étape 1: Charger le modèle pré-entraîné sur COCO (80 classes d'objets)
    # weights='DEFAULT' charge les meilleurs poids disponibles
    # Transfer Learning: On réutilise les connaissances du modèle (formes, textures, contours)
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Étape 2: Adapter la tête de classification à notre problème (21 classes)
    # On récupère le nombre de features d'entrée de la couche actuelle
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # On remplace la tête de prédiction COCO (80 classes) par notre tête personnalisée (21 classes)
    # FastRCNNPredictor crée une nouvelle couche de classification et de régression
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Retourner le modèle modifié, prêt pour l'entraînement
    return model

# Message de confirmation au chargement du module
print("✅ Architecture Faster R-CNN chargée (model.py)")
