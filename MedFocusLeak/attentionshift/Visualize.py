import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from surrogate.Blip import Blip2Extractor

# --- Configuration ---
IMAGE_DIR = "Imgs"
ATTACKED_DIR = "result"
VIS_DIR = "visualizations" # Directory to save the output plots

MODEL_NAME = "Salesforce/blip2-opt-2.7b" # Use the same model as the attack

# Visualization parameters
COLORMAP = 'viridis' # Colormap for the attention heatmaps
OVERLAY_ALPHA = 0.6 # Transparency of the heatmap overlay


def process_attention_map(attention_tensor, model_input_size, original_image_size):
    """
    Processes a raw attention tensor from the model into a grid suitable for visualization.
    """
    # We are interested in the attention from the [CLS] token to the image patch tokens.
    # Shape of attention_tensor: [batch_size, num_heads, seq_len, seq_len]
    # For vision transformers, seq_len is num_patches + 1 ([CLS] token)
    # We take the attention from the first token ([CLS]) to all other tokens (the patches).
    cls_attention = attention_tensor[0, :, 0, 1:].mean(dim=0) # Average across all heads

    # Determine the grid size (e.g., 14x14 or 16x16)
    num_patches = cls_attention.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError("Cannot form a square grid from the number of patches.")

    # Reshape the attention vector into a 2D grid
    attention_grid = cls_attention.reshape(grid_size, grid_size)

    # Resize the small attention grid to the size of the original image for overlay
    # We need to add batch and channel dimensions for interpolate
    attention_grid = attention_grid.unsqueeze(0).unsqueeze(0)
    resized_attention = F.interpolate(
        attention_grid,
        size=original_image_size,
        mode='bicubic',
        align_corners=False
    ).squeeze() # Remove batch and channel dims

    # Normalize the resized attention map to be between 0 and 1 for the colormap
    normalized_attention = (resized_attention - resized_attention.min()) / (resized_attention.max() - resized_attention.min())
    
    return normalized_attention.cpu().numpy()


def generate_visualization_for_image(extractor, original_img_path, attacked_img_path, output_path):
    """
    Generates and saves a single plot comparing attention for an original and attacked image.
    """
    try:
        original_image = Image.open(original_img_path).convert("RGB")
        attacked_image = Image.open(attacked_img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Skipping {os.path.basename(original_img_path)}: Attacked image not found at {attacked_img_path}")
        return

    print(f"Processing {os.path.basename(original_img_path)}...")

    # Get attention maps for both images
    # The extractor.forward() method returns a dictionary of detached, CPU-based tensors
    original_maps = extractor.forward(original_image)
    attacked_maps = extractor.forward(attacked_image)
    
    num_layers = len(original_maps)
    if num_layers == 0:
        print("Warning: No attention maps were extracted. Check model and hooks.")
        return

    # Create a tall plot with 4 columns: Original Img, Attacked Img, Original Attn, Attacked Attn
    fig, axes = plt.subplots(nrows=num_layers, ncols=4, figsize=(16, 4 * num_layers))
    fig.suptitle(f'Attention Comparison for {os.path.basename(original_img_path)}', fontsize=20, y=1.0)
    
    model_input_size = (
        extractor.processor.image_processor.size['height'],
        extractor.processor.image_processor.size['width']
    )

    for layer_idx in sorted(original_maps.keys()):
        ax_row = axes[layer_idx]

        # --- Plot Original and Attacked Images ---
        ax_row[0].imshow(original_image)
        ax_row[0].set_title(f"Layer {layer_idx}\nOriginal Image")
        ax_row[0].axis('off')

        ax_row[1].imshow(attacked_image)
        ax_row[1].set_title("Attacked Image")
        ax_row[1].axis('off')

        # --- Process and Plot Original Attention ---
        orig_attn_processed = process_attention_map(original_maps[layer_idx], model_input_size, original_image.size[::-1])
        ax_row[2].imshow(original_image)
        ax_row[2].imshow(orig_attn_processed, cmap=COLORMAP, alpha=OVERLAY_ALPHA)
        ax_row[2].set_title("Original Attention")
        ax_row[2].axis('off')
        
        # --- Process and Plot Attacked Attention ---
        attacked_attn_processed = process_attention_map(attacked_maps[layer_idx], model_input_size, attacked_image.size[::-1])
        ax_row[3].imshow(attacked_image)
        ax_row[3].imshow(attacked_attn_processed, cmap=COLORMAP, alpha=OVERLAY_ALPHA)
        ax_row[3].set_title("Attacked Attention")
        ax_row[3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.99]) # Adjust layout to make room for suptitle
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved visualization to {output_path}")


def main():
    """
    Main function to run the visualization process.
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU, which will be slow.")
    
    # Create output directory
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
        print(f"Created visualization directory: {VIS_DIR}")

    # Load model and extractor
    print("Initializing BLIP-2 model and extractor...")
    extractor = Blip2Extractor(model_name=MODEL_NAME)
    print("Model loaded.")
    
    try:
        image_files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    except FileNotFoundError:
        print(f"ERROR: Image directory not found at '{IMAGE_DIR}'. Please create it.")
        return
        
    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Exiting.")
        return

    # Process each image
    for image_name in tqdm(image_files, desc="Generating Visualizations"):
        original_path = os.path.join(IMAGE_DIR, image_name)
        attacked_path = os.path.join(ATTACKED_DIR, image_name)
        
        # Create a unique name for the output visualization file
        base_name, _ = os.path.splitext(image_name)
        output_path = os.path.join(VIS_DIR, f"attention_comparison_{base_name}.png")
        
        generate_visualization_for_image(extractor, original_path, attacked_path, output_path)

    print("\nAll visualizations have been generated.")


if __name__ == "__main__":
    main()