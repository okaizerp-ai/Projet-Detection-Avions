PROJET\_DETECTION\_AVIONS/
├── config.py              # Configuration centralisée (chemins, hyperparamètres, device)
├── train_V1.py            # Script d'entraînement avec gestion de checkpoints
├── train_V2.py            # Script d'entraînement avec gestion de checkpoints
├── predictions.json       # Fichier pour la validation du modèle
├── model.py               # Définition de l'architecture Faster R-CNN
├── dataset.py             # Classe PlaneDataset pour le chargement Images/XML (Pascal VOC)
├── app.py       	   # Interface graphique utilisateur (GUI)
│
├── data/                  # Dossier des données (Images et Annotations)
│   ├── train\_images/      # Images pour l'entraînement (V1, V2, V3)
│   ├── annotations/       # Fichiers XML correspondants
│   └── eval\_images/       # Images d'évaluation "aveugles" (fournies par le professeur)
│
├── models/                # Sauvegardes des poids du modèle (.pth)
│   ├── faster\_rcnn\_V1.pth     # Baseline (0.69 F1)
│   ├── faster\_rcnn\_V2.pth     # Fine-tuned (Augmentation de données)
│   └── faster\_rcnn\_ELITE.pth  # Version finale optimisée (WeightedSampler)
│
└── outputs/               # Résultats et exports
├── detections/        # Images traitées avec boîtes englobantes et labels
└── stats/             # Graphiques (Boxplots, Matrice de Confusion)



