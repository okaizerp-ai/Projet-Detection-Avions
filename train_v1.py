import torch
import os
import config  # Import de ton fichier de configuration
from model import get_model_instance_segmentation
from dataset import PlaneDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

def get_transform():
    return T.Compose([T.ToTensor()])

# --- CONFIGURATION VIA CONFIG.PY ---
DEVICE = config.DEVICE

# 1. Dataset et DataLoader
# On pointe vers le dossier data universel
dataset = PlaneDataset(config.DATA_DIR, transforms=get_transform())

# Logique de Subset : On garde 50 images pour le test final de la V1
indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:-50])

data_loader = DataLoader(
    dataset_train, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x))
)

# 2. Modèle (Architecture de base)
model = get_model_instance_segmentation(config.NUM_CLASSES)
model.to(DEVICE)

# 3. Optimiseur (LR à 0.005 pour le premier entraînement)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 4. Boucle d'entraînement
num_epochs = 10
print(f"🚀 Début de l'entraînement Baseline (V1) sur : {DEVICE}")

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Époque {epoch+1}, Itération {i}, Loss: {losses.item():.4f}")
        i += 1

# 5. Sauvegarde Universelle
# On l'enregistre dans le dossier 'models/' avec le nom attendu par le script V2
save_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions.pth')
torch.save(model.state_dict(), save_path)

print(f"✅ Entraînement V1 terminé ! Modèle sauvegardé dans : {save_path}")