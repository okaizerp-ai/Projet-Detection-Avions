import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm
import torchvision.transforms as T

# Import de tes fichiers locaux
import config 
from model import get_model_instance_segmentation
from dataset import PlaneDataset

def run_full_confusion_matrix():
    print(f"⚙️ Racine du projet : {config.ROOT_DIR}")
    print(f"🖥️ Calcul de la matrice sur : {config.DEVICE}")

    # 1. Chemins via config
    model_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
    output_path = os.path.join(config.OUTPUTS_DIR, 'matrice_confusion_TOTALE.png')

    # 2. Chargement du modèle Elite
    model = get_model_instance_segmentation(config.NUM_CLASSES)
    if not os.path.exists(model_path):
        print(f" Erreur : Modèle introuvable à {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()

    # 3. Préparation du Dataset COMPLET (1331 images)
    # On n'utilise plus de "Subset", on prend tout le dossier data/images
    dataset = PlaneDataset(config.DATA_DIR, transforms=T.Compose([T.ToTensor()]))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x))
    )

    all_preds = []
    all_gt = []

    print(f"🔍 Analyse complète de {len(dataset)} images en cours...")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(config.DEVICE) for img in images]
            outputs = model(images)
            
            for i in range(len(targets)):
                gt_labels = targets[i]['labels'].cpu().numpy()
                pred_labels = outputs[i]['labels'].cpu().numpy()
                pred_scores = outputs[i]['scores'].cpu().numpy()
                
                # On garde les prédictions avec une confiance > 0.4 
                # (un peu plus bas pour capter toutes les intentions de l'IA)
                mask = pred_scores > 0.4
                valid_preds = pred_labels[mask]
                
                if len(valid_preds) > 0 and len(gt_labels) > 0:
                    # On compare le premier avion détecté au premier avion réel de l'image
                    all_preds.append(valid_preds[0])
                    all_gt.append(gt_labels[0])

    # 4. Création de la Matrice (A1 à A20)
    labels_range = list(range(1, 21))
    cm = confusion_matrix(all_gt, all_preds, labels=labels_range)

    # Normalisation (%)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # 5. Rendu Visuel
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', # Vert pour changer
                xticklabels=config.CLASSES[1:], yticklabels=config.CLASSES[1:])
    
    plt.title(f'Matrice de Confusion Globale ({len(dataset)} images) - Modèle Elite', fontsize=18)
    plt.xlabel('Prédictions de l\'IA', fontsize=14)
    plt.ylabel('Vérité Terrain (Annotations XML)', fontsize=14)
    
    plt.savefig(output_path)
    plt.show()
    print(f" Matrice finale sauvegardée dans : {output_path}")

if __name__ == "__main__":
    run_full_confusion_matrix()