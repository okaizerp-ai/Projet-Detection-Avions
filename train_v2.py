import torch
import os
import config  # <--- Ton GPS universel
from model import get_model_instance_segmentation
from dataset import PlaneDataset
import torchvision.transforms as T
from tqdm import tqdm 

# --- CONFIGURATION VIA CONFIG.PY ---
DEVICE = config.DEVICE

# 1. Charger le modèle avec l'architecture définie (21 classes)
model = get_model_instance_segmentation(config.NUM_CLASSES)

# On cherche le poids initial (V1) dans le dossier 'models' défini dans config
checkpoint_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions.pth')

if os.path.exists(checkpoint_path):
    print(f"💎 Chargement du modèle existant (V1) depuis : {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
else:
    print(f"⚠️ Attention : {checkpoint_path} non trouvé. Le Fine-Tuning partira de zéro.")

model.to(DEVICE)

# 2. Optimiseur : Learning Rate fin (0.0005) pour ne pas détruire les acquis de la V1
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

# 3. DataLoader
# On utilise config.DATA_DIR qui pointe vers tes images et XML
dataset = PlaneDataset(config.DATA_DIR, transforms=T.Compose([T.ToTensor()]))
data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=0, # Plus stable sur tous les OS
    collate_fn=lambda x: tuple(zip(*x))
)

# 4. Entraînement d'amélioration (V2)
num_epochs = 7 
print(f"🚀 Début du Fine-Tuning V2 sur : {DEVICE}")

for epoch in range(num_epochs):
    model.train()
    prog_bar = tqdm(data_loader, desc=f"Époque {epoch+1}/{num_epochs}")
    
    epoch_loss = 0
    for images, targets in prog_bar:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        prog_bar.set_postfix(loss=losses.item())
    
    avg_loss = epoch_loss / len(data_loader)
    print(f"✅ Époque {epoch+1} terminée. Perte moyenne : {avg_loss:.4f}")

# 5. Sauvegarde Universelle du modèle V2
# Il sera enregistré dans le dossier 'models/' de ton projet
save_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
torch.save(model.state_dict(), save_path)

print(f"✨ Bravo ! Le modèle V2 est sauvegardé ici : {save_path}")