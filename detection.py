import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import config # Utilise ton GPS
from model import get_model_instance_segmentation
from tqdm import tqdm

# Configuration des dossiers via config
MODEL_PATH = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
OUTPUT_DIR = os.path.join(config.OUTPUTS_DIR, 'DETECTIONS_VISUELLES')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Chargement du modèle
model = get_model_instance_segmentation(config.NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
model.to(config.DEVICE).eval()

# 2. Liste des images
img_dir = config.EVAL_IMG
all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()

print(f"🎨 Génération des visuels dans : {OUTPUT_DIR}")

with torch.no_grad():
    for img_name in tqdm(all_images):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(config.DEVICE)
        
        prediction = model(img_tensor)
        draw = ImageDraw.Draw(img)
        
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] > 0.5:
                box = boxes[i]
                label_txt = f"{config.CLASSES[labels[i]]} ({int(scores[i]*100)}%)"
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="lime", width=4)
                draw.rectangle([(box[0], box[1] - 25), (box[0] + 130, box[1])], fill="lime")
                draw.text((box[0] + 5, box[1] - 22), label_txt, fill="black", font=font)

        img.save(os.path.join(OUTPUT_DIR, f"visu_{img_name}"))

print(f"\n✅ Terminé !")