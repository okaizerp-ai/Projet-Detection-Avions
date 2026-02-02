"""
dataset.py - Classe Dataset Personnalisée pour les Images d'Avions

Ce fichier définit la classe PlaneDataset qui hérite de torch.utils.data.Dataset.
Elle est responsable de:
1. Charger les images et leurs annotations XML (format PASCAL VOC)
2. Appliquer la data augmentation (flips horizontal et vertical)
3. Retourner les données au format attendu par Faster R-CNN

La data augmentation est CIBLÉE: les classes faibles (peu d'exemples) ont 80% 
de chances de flip, les autres 50%, pour équilibrer l'apprentissage.
"""

import torch  # PyTorch pour manipuler les tensors
from PIL import Image  # Bibliothèque pour ouvrir et manipuler les images
import os  # Manipulation des chemins de fichiers
import xml.etree.ElementTree as ET  # Parser XML pour lire les annotations PASCAL VOC
import random  # Génération de nombres aléatoires pour data augmentation
import torchvision.transforms.functional as F  # Transformations d'images (flip, rotation, etc.)
import config  # Import de la configuration centralisée (chemins, classes faibles)

class PlaneDataset(torch.utils.data.Dataset):
    """
    Dataset personnalisé pour charger les images d'avions et leurs annotations.
    
    Hérite de torch.utils.data.Dataset, ce qui permet de l'utiliser avec DataLoader.
    """
    
    def __init__(self, root, transforms=None):
        """
        Constructeur du dataset.
        
        Args:
            root (str): Chemin vers le dossier racine contenant /images et /annotations
            transforms: Transformations à appliquer (typiquement ToTensor())
        """
        self.root = root  # Stockage du chemin racine
        self.transforms = transforms  # Stockage des transformations
        
        # ========== CONSTRUCTION DES CHEMINS VERS LES SOUS-DOSSIERS ==========
        # os.path.join() assure la compatibilité multi-OS (/ vs \)
        self.imgs_dir = os.path.join(root, "images")  # Dossier des images JPG
        self.ann_dir = os.path.join(root, "annotations")  # Dossier des annotations XML
        
        # ========== CHARGEMENT ET TRI DES FICHIERS ==========
        # os.listdir() retourne tous les fichiers du dossier
        # sorted() trie par ordre alphabétique (CRUCIAL: imgs[0] doit correspondre à annotations[0])
        # list() convertit en liste Python
        self.imgs = list(sorted(os.listdir(self.imgs_dir)))
        self.annotations = list(sorted(os.listdir(self.ann_dir)))
        
        # ========== RÉCUPÉRATION DE LA LISTE DES CLASSES FAIBLES ==========
        # getattr(object, 'attribut', default) : récupère config.WEAK_CLASSES ou liste par défaut
        # Permet au code de fonctionner même si config.WEAK_CLASSES n'existe pas
        self.weak_classes = getattr(config, 'WEAK_CLASSES', [1, 12, 15, 18, 20])

    def __getitem__(self, idx):
        """
        Méthode appelée par PyTorch pour récupérer un élément du dataset.
        
        Args:
            idx (int): Index de l'image à récupérer (0 à len(dataset)-1)
        
        Returns:
            tuple: (image_tensor, target_dict)
                - image_tensor: Image en format tensor (C, H, W)
                - target_dict: Dictionnaire contenant boxes, labels, area, etc.
        """
        
        # ========== CONSTRUCTION DES CHEMINS COMPLETS ==========
        # Combine le chemin du dossier + nom du fichier
        img_path = os.path.join(self.imgs_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])
        
        # ========== CHARGEMENT DE L'IMAGE ==========
        # Image.open() ouvre l'image (JPG, PNG, etc.)
        # .convert("RGB") force le format RGB (3 canaux) même si l'image est en niveaux de gris
        img = Image.open(img_path).convert("RGB")
        
        # Récupération de la taille de l'image (nécessaire pour ajuster les boîtes après flip)
        width, height = img.size  # Tuple (largeur, hauteur) en pixels
        
        # ========== PARSING DU FICHIER XML ==========
        # ET.parse() lit et parse le fichier XML
        tree = ET.parse(ann_path)
        # .getroot() retourne l'élément racine du XML (balise <annotation>)
        root = tree.getroot()
        
        # Initialisation des listes pour stocker les boîtes et labels
        boxes = []  # Liste des bounding boxes [[xmin, ymin, xmax, ymax], ...]
        labels = []  # Liste des labels (indices de classe) [5, 12, 3, ...]
        
        # Flag pour détecter si l'image contient une classe faible
        has_weak_class = False
        
        # ========== EXTRACTION DES ANNOTATIONS ==========
        # root.findall('object') trouve toutes les balises <object> (un par avion)
        for obj in root.findall('object'):
            # Extraction du nom de la classe (ex: "A15")
            # obj.find('name') trouve la balise <name>, .text récupère le contenu texte
            label_name = obj.find('name').text
            
            # Conversion du label texte en index numérique
            # label_name[1:] extrait les chiffres après 'A' (ex: "A15" → "15")
            # int() convertit la chaîne en entier (ex: "15" → 15)
            label_idx = int(label_name[1:])
            
            # Vérification si cette classe est dans la liste des classes faibles
            if label_idx in self.weak_classes:
                has_weak_class = True  # Active le flag pour augmentation agressive
                
            # ========== EXTRACTION DES COORDONNÉES DE LA BOUNDING BOX ==========
            # Trouve la balise <bndbox> qui contient les coordonnées
            xmlbox = obj.find('bndbox')
            
            # Extraction des 4 coordonnées et conversion en float
            # .find('xmin').text récupère le contenu texte de la balise <xmin>
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # ========== SÉCURITÉ CONTRE LES BOÎTES INVERSÉES ==========
            # Certaines annotations peuvent avoir xmin > xmax (erreur humaine)
            # min/max garantit que xmin < xmax et ymin < ymax
            boxes.append([min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)])
            
            # Ajout du label (index de classe) à la liste
            labels.append(label_idx)

        # ========== DATA AUGMENTATION "ELITE" (CIBLÉE PAR CLASSE) ==========
        # Probabilité de transformation dépend de la présence d'une classe faible
        # Classes faibles: 80% de chance de flip (boost pour compenser le manque de données)
        # Classes normales: 50% de chance de flip (augmentation standard)
        prob = 0.8 if has_weak_class else 0.5
        
        # ========== FLIP HORIZONTAL (SYMÉTRIE VERTICALE) ==========
        # random.random() génère un float aléatoire entre 0 et 1
        # Si ce nombre < prob, on applique le flip
        if random.random() < prob:
            # F.hflip() retourne l'image inversée horizontalement
            img = F.hflip(img)
            
            # CRUCIAL: Ajuster les coordonnées des boîtes après le flip
            # Le coin gauche devient le coin droit et vice-versa
            new_boxes = []
            for box in boxes:
                # xmin_new = width - xmax_old (le bord droit devient le bord gauche)
                # xmax_new = width - xmin_old (le bord gauche devient le bord droit)
                # ymin et ymax ne changent pas (flip horizontal n'affecte pas la hauteur)
                new_boxes.append([width - box[2], box[1], width - box[0], box[3]])
            boxes = new_boxes  # Remplacement de l'ancienne liste

        # ========== FLIP VERTICAL (SYMÉTRIE HORIZONTALE) ==========
        # Même logique que le flip horizontal mais sur l'axe vertical
        if random.random() < prob:
            # F.vflip() retourne l'image inversée verticalement
            img = F.vflip(img)
            
            # Ajustement des coordonnées Y après le flip
            new_boxes = []
            for box in boxes:
                # ymin_new = height - ymax_old (le haut devient le bas)
                # ymax_new = height - ymin_old (le bas devient le haut)
                # xmin et xmax ne changent pas (flip vertical n'affecte pas la largeur)
                new_boxes.append([box[0], height - box[3], box[2], height - box[1]])
            boxes = new_boxes

        # ========== CONVERSION EN TENSORS PYTORCH ==========
        # torch.as_tensor() convertit une liste Python en tensor PyTorch
        # dtype=torch.float32 : Type float 32 bits pour les coordonnées (précision standard)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # dtype=torch.int64 : Type int 64 bits pour les labels (indices de classe)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # ========== CONSTRUCTION DU DICTIONNAIRE TARGET ==========
        # Format attendu par Faster R-CNN pour l'entraînement
        target = {}
        
        # Boîtes englobantes au format [xmin, ymin, xmax, ymax]
        target["boxes"] = boxes
        
        # Labels (indices de classe: 1 à 20 pour A1 à A20)
        target["labels"] = labels
        
        # ID de l'image (utilisé par PyTorch pour suivi)
        target["image_id"] = torch.tensor([idx])
        
        # ========== CALCUL DE LA SURFACE DES BOÎTES ==========
        # boxes[:, 3] = colonne 3 (ymax) de toutes les lignes
        # boxes[:, 1] = colonne 1 (ymin) de toutes les lignes
        # Formule: area = (ymax - ymin) * (xmax - xmin)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # ========== FLAG ISCROWD ==========
        # Tensor de zéros indiquant qu'aucun objet n'est groupé/crowd
        # iscrowd=1 signifierait que plusieurs objets sont fusionnés en une seule annotation
        # len(labels) = nombre d'objets dans l'image
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        # ========== APPLICATION DES TRANSFORMATIONS ==========
        # Si des transformations ont été passées au constructeur (typiquement ToTensor())
        if self.transforms is not None:
            # Application des transformations à l'image
            # ToTensor() convertit PIL Image (H, W, C) en tensor (C, H, W) et normalise [0-1]
            img = self.transforms(img)

        # Retour du tuple (image, annotations) attendu par PyTorch
        return img, target

    def __len__(self):
        """
        Retourne le nombre total d'images dans le dataset.
        
        Méthode obligatoire pour une classe Dataset.
        Permet à PyTorch de savoir combien d'éléments peuvent être récupérés.
        
        Returns:
            int: Nombre d'images dans le dataset
        """
        return len(self.imgs)
