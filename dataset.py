import torch
from PIL import Image
import os
import xml.etree.ElementTree as ET
import random
import torchvision.transforms.functional as F
import config # Import de ton GPS universel

class PlaneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        """
        Args:
            root (str): Chemin vers le dossier racine des données (contenant /images et /annotations)
            transforms: Transformations à appliquer
        """
        self.root = root
        self.transforms = transforms
        
        # Chemins universels vers les sous-dossiers
        self.imgs_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        
        # Chargement et tri des fichiers
        self.imgs = list(sorted(os.listdir(self.imgs_dir)))
        self.annotations = list(sorted(os.listdir(self.ann_dir)))
        
        # On récupère la liste des classes à booster depuis config.py
        # Si config.WEAK_CLASSES n'existe pas, on prend ta liste par défaut
        self.weak_classes = getattr(config, 'WEAK_CLASSES', [1, 12, 15, 18, 20])

    def __getitem__(self, idx):
        # Chemins complets
        img_path = os.path.join(self.imgs_dir, self.imgs[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # Détection des classes présentes dans l'image
        has_weak_class = False
        for obj in root.findall('object'):
            label_name = obj.find('name').text # ex: "A15"
            label_idx = int(label_name[1:]) # On garde l'index (15)
            
            if label_idx in self.weak_classes:
                has_weak_class = True
                
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # Sécurité pour éviter les boîtes inversées
            boxes.append([min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)])
            labels.append(label_idx)

        # --- DATA AUGMENTATION "ELITE" (Box-aware) ---
        # Si l'avion est "difficile", on augmente la probabilité de transformation
        prob = 0.8 if has_weak_class else 0.5
        
        # Flip Horizontal
        if random.random() < prob:
            img = F.hflip(img)
            new_boxes = []
            for box in boxes:
                new_boxes.append([width - box[2], box[1], width - box[0], box[3]])
            boxes = new_boxes

        # Flip Vertical
        if random.random() < prob:
            img = F.vflip(img)
            new_boxes = []
            for box in boxes:
                new_boxes.append([box[0], height - box[3], box[2], height - box[1]])
            boxes = new_boxes

        # Conversion finale en tenseurs
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)