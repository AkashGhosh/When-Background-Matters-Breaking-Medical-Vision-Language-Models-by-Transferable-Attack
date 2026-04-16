import os
import torch
from PIL import Image
from tqdm import tqdm

# Import the necessary classes from your files
from surrogate.Blip import Blip2Extractor
from Attack import AttentionPerturber

# --- Configuration ---
# --- IMPORTANT: Set your folder paths here ---
IMAGE_DIR = "Imgs"
MASK_DIR = "masks"
RESULT_DIR = "result"

# --- Model and Attack Parameters ---
# Salesforce/blip2-opt-2.7b has 32 vision layers (0-31)
MODEL_NAME = "Salesforce/blip2-opt-2.7b" 
# Specify which layers of the vision encoder to attack. None means all layers.
# TARGET_LAYERS = [30, 31]

# PGD attack parameters
NUM_STEPS = 500
# Perturbation budget and step size
# 8/255 is a common setting for epsilon
EPSILON = 16/ 255.0
ALPHA = 4/ 255.0


def run_attack():
    """
    Main function to run the adversarial attack on a directory of images.
    """
    # --- Setup ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU to run.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"Created output directory: {RESULT_DIR}")

    # --- Model Initialization ---
    print("Initializing BLIP-2 model and extractor...")
    # The extractor will load the model onto the GPU
    try:
        extractor = Blip2Extractor(model_name=MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model. Please ensure you have an internet connection and necessary libraries. Error: {e}")
        return
    print("Model loaded successfully.")

    # --- Processing Loop ---
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    except FileNotFoundError:
        print(f"ERROR: Image directory not found at '{IMAGE_DIR}'. Please create it and add images.")
        return
        
    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Exiting.")
        return

    print(f"\nFound {len(image_files)} images to process in '{IMAGE_DIR}'. Starting attack loop...")

    for image_name in tqdm(image_files, desc="Processing Images"):
        print(f"\n--- Attacking {image_name} ---")
        try:
            # --- Load Data ---
            image_path = os.path.join(IMAGE_DIR, image_name)
            # Assume mask has the same name but could have a different extension (e.g., .png)
            mask_name, _ = os.path.splitext(image_name)
            mask_path = os.path.join(MASK_DIR, mask_name + ".png") # Common for masks to be PNG
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(MASK_DIR, image_name) # Try original name
                 if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {image_name} at '{mask_path}'. Skipping.")
                    continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)

            # --- STEP 1: Generate Original Attention Maps Beforehand ---
            print("Generating original attention maps for the clean image...")
            # The extractor's forward pass takes a PIL image and returns a dictionary of attention tensors.
            # This is run under torch.no_grad() inside the extractor.
            original_attention_maps = extractor.forward(image)
            
            # --- STEP 2: Instantiate the Attacker with Pre-calculated Maps ---
            print("Initializing the attacker with the original attention maps...")
            attacker = AttentionPerturber(
                extractor=extractor,
                image=image,
                mask=mask,
                attention_layers=original_attention_maps, # Pass the pre-calculated maps here
            )

            # --- STEP 3: Perform the Attack ---
            perturbed_image = attacker.perturb(
                num_steps=NUM_STEPS,
                alpha=ALPHA,
                epsilon=EPSILON
            )

            # --- STEP 4: Save the Result ---
            output_path = os.path.join(RESULT_DIR, image_name)
            perturbed_image.save(output_path)
            print(f"Successfully saved attacked image to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n--- All images processed. ---")


if __name__ == "__main__":
    run_attack()