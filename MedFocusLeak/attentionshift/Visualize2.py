import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Import the model factory from our __init__.py
from surrogate import MODEL_CLASS_MAP

# --- Configuration ---
IMAGE_DIR = "Imgs"
ATTACKED_DIR = "result"
VIS_DIR = "visualizations"

# --- KEY CHANGE: Define the same ensemble used for the attack ---
ENSEMBLE_MODELS = [
    "Salesforce/blip2-opt-2.7b",
    "openai/clip-vit-base-patch16",
    "Salesforce/blip-itm-base-coco",
]

# Visualization parameters
COLORMAP = 'viridis'
OVERLAY_ALPHA = 0.6

def get_safe_model_name(hf_name):
    """Creates a filesystem-safe name from a Hugging Face model name."""
    return hf_name.replace("/", "_")

def process_attention_map(attention_tensor, original_image_size):
    cls_attention = attention_tensor[0, :, 0, 1:].mean(dim=0)
    num_patches = cls_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot form a square grid from {num_patches} patches.")

    attention_grid = cls_attention.reshape(grid_size, grid_size)
    attention_grid = attention_grid.unsqueeze(0).unsqueeze(0)
    resized_attention = F.interpolate(
        attention_grid,
        size=original_image_size,
        mode='bicubic',
        align_corners=False
    ).squeeze()
    
    # Check for non-finite values before normalization
    if not torch.all(torch.isfinite(resized_attention)):
        return np.zeros(original_image_size) # Return a black map on failure

    min_val, max_val = resized_attention.min(), resized_attention.max()
    if max_val - min_val < 1e-6: # Avoid division by zero
        return np.zeros(original_image_size)

    normalized_attention = (resized_attention - min_val) / (max_val - min_val)
    return normalized_attention.cpu().numpy()


def generate_visualization_for_image_and_model(extractor, original_img, attacked_img, vis_output_path):
    """
    Generates and saves a plot for a single model in the ensemble.
    """
    model_name = get_safe_model_name(extractor.model.name_or_path)
    print(f"  - Visualizing for model: {model_name}")

    original_maps = extractor.forward(original_img)
    attacked_maps = extractor.forward(attacked_img)
    
    num_layers = len(original_maps)
    if num_layers == 0:
        print(f"  - Warning: No attention maps extracted for model {model_name}.")
        return

    fig, axes = plt.subplots(nrows=num_layers, ncols=4, figsize=(16, 4 * num_layers), squeeze=False)
    fig.suptitle(f'Attention Comparison for {model_name}', fontsize=20, y=1.0)
    
    for layer_idx in sorted(original_maps.keys()):
        ax_row = axes[layer_idx, :]

        ax_row[0].imshow(original_img)
        ax_row[0].set_title(f"Layer {layer_idx}\nOriginal Image")
        ax_row[0].axis('off')

        ax_row[1].imshow(attacked_img)
        ax_row[1].set_title("Attacked Image")
        ax_row[1].axis('off')

        orig_attn_processed = process_attention_map(original_maps[layer_idx], original_img.size[::-1])
        ax_row[2].imshow(original_img)
        ax_row[2].imshow(orig_attn_processed, cmap=COLORMAP, alpha=OVERLAY_ALPHA)
        ax_row[2].set_title("Original Attention")
        ax_row[2].axis('off')
        
        attacked_attn_processed = process_attention_map(attacked_maps[layer_idx], attacked_img.size[::-1])
        ax_row[3].imshow(attacked_img)
        ax_row[3].imshow(attacked_attn_processed, cmap=COLORMAP, alpha=OVERLAY_ALPHA)
        ax_row[3].set_title("Attacked Attention")
        ax_row[3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(vis_output_path)
    plt.close(fig)
    print(f"  - Saved visualization to {vis_output_path}")


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU, which will be slow.")
    
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
        print(f"Created visualization directory: {VIS_DIR}")

    # --- Load the entire ensemble of models ---
    print("Initializing ensemble of surrogate models for visualization...")
    extractors = [MODEL_CLASS_MAP[name]() for name in ENSEMBLE_MODELS if name in MODEL_CLASS_MAP]
    if not extractors:
        print("No models loaded. Exiting.")
        return
    print(f"Loaded {len(extractors)} models.")
    
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    except FileNotFoundError:
        print(f"ERROR: Image directory not found at '{IMAGE_DIR}'.")
        return
        
    # Process each image
    for image_name in tqdm(image_files, desc="Generating Visualizations"):
        original_path = os.path.join(IMAGE_DIR, image_name)
        attacked_path = os.path.join(ATTACKED_DIR, image_name)
        
        try:
            original_image = Image.open(original_path).convert("RGB")
            attacked_image = Image.open(attacked_path).convert("RGB")
        except FileNotFoundError:
            print(f"Skipping {image_name}: Attacked image not found at {attacked_path}")
            continue

        print(f"\nProcessing {image_name}...")
        
        # --- Generate a separate visualization for EACH model in the ensemble ---
        for extractor in extractors:
            base_name, _ = os.path.splitext(image_name)
            model_safe_name = get_safe_model_name(extractor.model.name_or_path)
            output_path = os.path.join(VIS_DIR, f"attention_{base_name}_{model_safe_name}.png")
            
            generate_visualization_for_image_and_model(
                extractor, original_image, attacked_image, output_path
            )

    # Clean up by removing extractors and clearing GPU memory
    del extractors
    torch.cuda.empty_cache()
    print("\nAll visualizations have been generated.")


if __name__ == "__main__":
    main()