import os
import torch

# 1. Détection automatique de la racine du projet
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Dossiers principaux
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')

# Sous-dossiers de données
# Images d'entraînement et annotations XML doivent être dans data/images et data/annotations
BASE_DATA_PATH = DATA_DIR 
# Dossier spécifique pour les images du prof
EVAL_IMG_DIR = os.path.join(DATA_DIR, 'eval_images', 'eval-dataset', 'images')

# 3. Paramètres du Modèle
NUM_CLASSES = 21  # 20 avions + 1 background
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 4. Définition des classes (Indispensable pour l'affichage et le JSON)
CLASSES = ['Background'] + [f'A{i}' for i in range(1, 21)]

# 5. Liste des classes à booster (Data Augmentation ciblée)
WEAK_CLASSES = [1, 12, 15, 18, 20] # A1, A12, A15, A18, A20

# 6. Création automatique des dossiers s'ils manquent
for d in [MODELS_DIR, OUTPUTS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"⚙️ Configuration chargée. Racine du projet : {ROOT_DIR}")