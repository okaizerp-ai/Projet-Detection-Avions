# Projet IA Détecteur d'Avions militaires 

## Structure 

le dossier contient : 

 - un fichier app.py (Interface graphique utilisateur (GUI))
 - un fichier config.py (Configuration centralisée (chemins, hyperparamètres, device))
 - un fichier dataset.py (Classe Dataset pour le chargement Images/XML (Pascal VOC))
 - un fichier detection.py ( création bounding boxes et de la classification A1 .... A20)
 - un fichier model.py  (Définition de l'architecture Faster R-CNN)
 - un fichier predictions.json (Fichier pour la validation du modèle)
 - un fichier train_v1.py (Script d'entraînement avec gestion de checkpoints)
 - un fichier train_v2.py  (Script d'entraînement supplémentaire au premier entraitement V1)


## Informations 

- le dossier dataset avec les annotations et les images étant trop lourd , ils ne seront pas pris en compte
- Quant au modèle pth, il ne sera pas pris en compte également. 

