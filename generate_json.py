import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
import config  # Utilise ton GPS
from model import get_model_instance_segmentation
from tqdm import tqdm

def run_evaluation():
    print(f" Inférence universelle sur : {config.DEVICE}")
    
    # 1. Chargement du modèle depuis le dossier 'models/'
    model = get_model_instance_segmentation(config.NUM_CLASSES)
    model_path = os.path.join(config.MODELS_DIR, 'faster_rcnn_avions_V2.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable dans : {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE).eval()

    # 2. Liste des images (Dossier prof défini dans config)
    eval_dir = config.EVAL_IMG
    if not os.path.exists(eval_dir):
        print(f"❌ Dossier images introuvable : {eval_dir}")
        return

    all_images = [f for f in os.listdir(eval_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = {}

    print(f"🔍 Analyse de {len(all_images)} images pour le JSON...")

    # 3. Détection
    with torch.no_grad():
        for img_name in tqdm(all_images):
            img_path = os.path.join(eval_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_t = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(config.DEVICE)
            
            prediction = model(img_t)
            
            img_preds = []
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] > 0.5:
                    box = boxes[i]
                    img_preds.append({
                        "class": config.CLASSES[labels[i]],
                        "score": float(scores[i]),
                        "coordinates": {
                            "xmin": int(box[0]), "ymin": int(box[1]),
                            "xmax": int(box[2]), "ymax": int(box[3])
                        }
                    })
            results[img_name] = img_preds

    # 4. Sauvegarde dans le dossier 'outputs/'
    output_json = os.path.join(config.OUTPUTS_DIR, 'predictions_officielles.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ JSON généré avec succès dans : {output_json}")

if __name__ == "__main__":
    run_evaluation()