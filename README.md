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
 - un fichier confusion.py   (création de la matrice de confusion)
 - un fichier generate_json.py  (générer le fichier json)



## Informations 

Pour des raisons de limitations de taille sur GitHub, certains fichiers essentiels ne sont pas inclus dans ce dépôt.
Vous devez les ajouter manuellement pour que le projet fonctionne.

1. Le Modèle (Poids)
   
- Créez un dossier nommé models/ à la racine.

- Placez-y votre fichier de poids entraîné (ex: faster_rcnn_avions_V2.pth).

2. Le Dataset

- Créez un dossier nommé data/ à la racine.

- Ce dossier doit contenir les sous-dossiers : annotations/, images/ et images_eval/.

## Important 

Pour garantir le bon fonctionnement du programme,
il est impératif de vérifier que les chemins d'accès vers vos dossiers de données (data) et de modèles (models) sont correctement configurés dans les fichiers du projet avant toute exécution.
