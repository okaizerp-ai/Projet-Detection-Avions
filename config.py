"""
config.py - Configuration Centralisée du Projet

Ce fichier centralise tous les paramètres et chemins du projet.
Il permet de rendre le code universel : peu importe où le projet est installé,
les chemins s'adaptent automatiquement grâce à la détection de la racine.

Avantages:
- Modification facile des paramètres (un seul fichier à éditer)
- Compatibilité multi-OS (Windows, Mac, Linux)
- Évite les chemins hardcodés (ex: "C:/Users/...")
"""

import os  # Manipulation des chemins de fichiers et création de dossiers
import torch  # PyTorch pour détection du GPU

# ========== 1. DÉTECTION AUTOMATIQUE DE LA RACINE DU PROJET ==========
# __file__ : Variable spéciale Python = chemin de ce fichier (config.py)
# os.path.abspath() : Convertit en chemin absolu (ex: /home/user/projet/config.py)
# os.path.dirname() : Récupère le dossier parent (ex: /home/user/projet)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR pointe maintenant vers la racine du projet, peu importe où il est installé

# ========== 2. DÉFINITION DES DOSSIERS PRINCIPAUX ==========
# os.path.join() : Combine des chemins proprement (gère / vs \ automatiquement)
# Permet de créer des chemins compatibles Windows (C:\projet\data) et Linux (/home/projet/data)

# Dossier data/ : contient les images et annotations XML
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Dossier models/ : contient les fichiers .pth (poids des modèles entraînés)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Dossier outputs/ : contient les résultats (JSON, images annotées, graphiques)
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')

# ========== 3. SOUS-DOSSIERS DE DONNÉES ==========
# Alias pour DATA_DIR (utilisé dans certains scripts pour compatibilité)
BASE_DATA_PATH = DATA_DIR 

# Chemin spécifique vers les images d'évaluation du professeur
# Structure exacte attendue: data/eval_images/eval-dataset/images/
EVAL_IMG_DIR = os.path.join(DATA_DIR, 'eval_images', 'eval-dataset', 'images')

# ========== 4. PARAMÈTRES DU MODÈLE ==========
# Nombre total de classes à détecter
# 20 types d'avions (A1 à A20) + 1 classe "Background" (fond/rien)
NUM_CLASSES = 21

# Détection automatique du device (GPU ou CPU)
# torch.cuda.is_available() retourne True si un GPU NVIDIA est détecté
# Si GPU disponible → device = 'cuda' (entraînement 10-50x plus rapide)
# Sinon → device = 'cpu' (plus lent mais fonctionne partout)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ========== 5. DÉFINITION DES CLASSES D'AVIONS ==========
# Liste de 21 éléments: ['Background', 'A1', 'A2', ..., 'A20']
# Utilisée pour convertir un index numérique (ex: 5) en label texte (ex: "A5")
# List comprehension: [f'A{i}' for i in range(1, 21)] génère ['A1', 'A2', ..., 'A20']
CLASSES = ['Background'] + [f'A{i}' for i in range(1, 21)]

# ========== 6. CLASSES FAIBLES (POUR DATA AUGMENTATION CIBLÉE) ==========
# Liste des classes sous-représentées dans le dataset
# Ces classes nécessitent un boost dans la data augmentation:
# - A1, A12, A15, A18, A20 : peu d'exemples dans le dataset d'entraînement
# Utilisée dans dataset.py pour appliquer 80% de flip au lieu de 50%
WEAK_CLASSES = [1, 12, 15, 18, 20] # A1, A12, A15, A18, A20

# ========== 7. CRÉATION AUTOMATIQUE DES DOSSIERS ==========
# Boucle sur les 3 dossiers principaux
for d in [MODELS_DIR, OUTPUTS_DIR, DATA_DIR]:
    # os.makedirs() crée le dossier s'il n'existe pas
    # exist_ok=True évite une erreur si le dossier existe déjà
    # Utile au premier lancement du projet sur une nouvelle machine
    os.makedirs(d, exist_ok=True)

# Message de confirmation au chargement du module
# Affiche le chemin de la racine pour vérifier que la configuration est correcte
print(f"⚙️ Configuration chargée. Racine du projet : {ROOT_DIR}")
