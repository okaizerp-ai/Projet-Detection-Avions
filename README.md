PROJET_DETECTION_AVIONS/
├── config.py              # Configuration centralisée (chemins, hyperparamètres, device)
├── train_V1.py            # Script d'entraînement avec gestion de checkpoints
├── train_V2.py            # Script d'entraînement avec gestion de checkpoints
├── model.py               # Définition de l'architecture Faster R-CNN
├── dataset.py             # Classe PlaneDataset pour le chargement Images/XML (Pascal VOC)
├── app.py                 # Interface graphique utilisateur (GUI)
├── prediction.json         # Fichier pour la validation du modèle
│
├── data/                  # Dossier des données (Images et Annotations)
│   ├── train_images/      # Images pour l'entraînement (V1, V2, V3)
│   ├── annotations/       # Fichiers XML correspondants
│   └── eval_images/       # Images d'évaluation "aveugles" (fournies par le professeur)
│
├── models/                # Sauvegardes des poids du modèle (.pth)
│   ├── faster_rcnn_V1.pth     # Baseline (0.69 F1)
│   └── faster_rcnn_V2.pth     # Fine-tuned (Augmentation de données)
│   
│
└── outputs/               # Résultats et exports
├── detections/        # Images traitées avec boîtes englobantes et labels
└── stats/             # Graphiques (Boxplots, Matrice de Confusion)
