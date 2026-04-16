# --- START OF FILE run_attack.py ---

import os
import torch
from PIL import Image
from tqdm import tqdm

from surrogate import MODEL_CLASS_MAP
from Attack2 import AttentionPerturber

# --- Configuration ---
IMAGE_DIR = "" #Put the path for Source Image Directory
MASK_DIR = ""  #Put the path for Mask Directory
RESULT_DIR = "" #Put the path for the Directory to Save the Results

# --- PGD attack parameters ---
NUM_STEPS = 500
EPSILON = 16/255.0
ALPHA = 1/255.0

# --- Manual Model Configuration Dictionary ---
ENSEMBLE_CONFIG = {
    # "Salesforce/blip2-opt-2.7b": {
    #     "class": MODEL_CLASS_MAP["Salesforce/blip2-opt-2.7b"], "input_size": (224, 224), "patch_size": 14,
    #     "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    # },
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K": {
        "class": MODEL_CLASS_MAP["laion/CLIP-ViT-G-14-laion2B-s12B-b42K"], "input_size": (224, 224), "patch_size": 14,
        "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    "openai/clip-vit-large-patch14-336": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-large-patch14-336"], "input_size": (336, 336), "patch_size": 14,
        "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    # "Salesforce/blip-itm-base-coco": {
    #     "class": MODEL_CLASS_MAP["Salesforce/blip-itm-base-coco"], "input_size": (384, 384), "patch_size": 16,
    #     "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    # },
    "openai/clip-vit-base-patch16": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-base-patch16"], "input_size": (224, 224), "patch_size": 16,
        "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    "openai/clip-vit-base-patch32": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-base-patch32"], "input_size": (224, 224), "patch_size": 32,
        "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
}

def run_attack():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # --- KEY CHANGE: Load all models ONCE before the loop ---
    print("Loading ensemble models... This will take a while and use significant VRAM.")
    loaded_extractors = []
    try:
        for model_name, config in ENSEMBLE_CONFIG.items():
            print(f" - Loading {model_name}...")
            extractor_class = config["class"]
            extractor = extractor_class(model_name=model_name)
            # Attach the manual config to the extractor object so it can be accessed later
            extractor.manual_config = config
            loaded_extractors.append(extractor)
    except Exception as e:
        print(f"Failed to load models. Error: {e}")
        # Clean up any partially loaded models
        del loaded_extractors
        torch.cuda.empty_cache()
        return

    print(f"\nSuccessfully loaded {len(loaded_extractors)} models into memory.")

    # --- Get image files ---
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"ERROR: Image directory not found at '{IMAGE_DIR}'.")
        return
    
    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Exiting.")
        return

    # --- Processing Loop ---
    print(f"\nFound {len(image_files)} images to process. Starting attack loop...")
    for image_name in tqdm(image_files, desc="Processing Images"):
        print(f"\n--- Attacking {image_name} ---")
        try:
            image_path = os.path.join(IMAGE_DIR, image_name)
            mask_name, _ = os.path.splitext(image_name)
            mask_path = os.path.join(MASK_DIR, mask_name + ".png")
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(MASK_DIR, image_name)
                 if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {image_name}. Skipping.")
                    continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            
            # --- KEY CHANGE: Pass the PRE-LOADED extractors to the attacker ---
            attacker = AttentionPerturber(
                extractors=loaded_extractors,
                image=image,
                mask=mask,
            )

            perturbed_image = attacker.perturb(
                num_steps=NUM_STEPS, alpha=ALPHA, epsilon=EPSILON
            )

            output_path = os.path.join(RESULT_DIR, image_name)
            perturbed_image.save(output_path)
            print(f"Successfully saved attacked image to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # --- KEY CHANGE: Clean up models at the very end ---
    print("\n--- All images processed. Cleaning up models from memory. ---")
    del loaded_extractors
    torch.cuda.empty_cache()
    print("Cleanup complete.")

if __name__ == "__main__":
    run_attack()