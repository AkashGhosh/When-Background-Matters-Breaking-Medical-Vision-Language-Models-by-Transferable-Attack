import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
import random

class TopKBackgroundPatch:
    def __init__(self, k=5, non_overlapping=True, min_patch_size=32):
        self.k = k
        self.overlap = non_overlapping
        self.min_patch_size = min_patch_size
        
    def __call__(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """
        Process tensor inputs instead of PIL images
        
        Args:
            image_tensor: Input image tensor of shape (C, H, W)
            mask_tensor: Mask tensor of shape (C, H, W) or (H, W)
            
        Returns:
            Tuple of (cropped_images, crop_coordinates)
        """
        # Convert mask to numpy for processing
        if mask_tensor.dim() == 3:
            # If mask has channels, take first channel
            mask_np = mask_tensor[0].cpu().numpy()
        else:
            mask_np = mask_tensor.cpu().numpy()
            
        # Normalize mask to 0-255 range if needed
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)
        
        # Debug: Print mask statistics
        print(f"Mask shape: {mask_np.shape}")
        print(f"Mask min: {mask_np.min()}, max: {mask_np.max()}")
        print(f"Unique values: {np.unique(mask_np)}")
        
        # Try both interpretations of the mask
        # First try: assume background (valid regions) are dark (< 128)
        binary_mask1 = (mask_np < 128).astype(np.uint8)
        patches1 = self._find_patches(binary_mask1)
        
        # Second try: assume background (valid regions) are bright (>= 128)
        binary_mask2 = (mask_np < 128).astype(np.uint8)
        patches2 = self._find_patches(binary_mask2)
        
        # Choose the interpretation that gives more/better patches
        if len(patches1) >= len(patches2):
            patches = patches1
            binary_mask = binary_mask1
            print("Using interpretation: background < 128")
        else:
            patches = patches2
            binary_mask = binary_mask2
            print("Using interpretation: background >= 128")
            
        print(f"Valid background pixels: {np.sum(binary_mask)}")
        print(f"Total patches found: {len(patches)}")
        
        if len(patches) == 0:
            return [], []
        
        # Filter patches by minimum size
        patches = [p for p in patches if p[0] >= self.min_patch_size]
        print(f"Patches after size filtering (min_size={self.min_patch_size}): {len(patches)}")
        
        if len(patches) == 0:
            return [], []
            
        # Sort patches by size (largest first)
        patches.sort(reverse=True, key=lambda x: x[0])
        
        # Select non-overlapping patches
        selected = self._select_non_overlapping_patches(patches, binary_mask.shape)
        
        print(f"Selected patches: {len(selected)}")
        for i, (top, left, size) in enumerate(selected):
            print(f"  Patch {i+1}: top={top}, left={left}, size={size}")
        
        # Randomly select one patch from the selected patches
        if len(selected) > 0:
            chosen_patch = random.choice(selected)
            # Crop using tensor operations
            top, left, size = chosen_patch
            cropped_tensor = image_tensor[:, top:top+size, left:left+size]
            return [cropped_tensor], [chosen_patch]
        else:
            return [], []
    
    def _find_patches(self, binary_mask):
        """Find all possible square patches using corrected dynamic programming"""
        H, W = binary_mask.shape
        
        # Initialize DP table
        dp = np.zeros((H, W), dtype=int)
        patches = []
        
        # Fill DP table
        for i in range(H):
            for j in range(W):
                if binary_mask[i, j] == 1:  # Valid background pixel
                    if i == 0 or j == 0:
                        dp[i, j] = 1
                    else:
                        dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + 1
                    
                    # For each possible square size from 1 to dp[i,j]
                    for size in range(1, dp[i, j] + 1):
                        top = i - size + 1
                        left = j - size + 1
                        
                        # Verify this is actually a valid square
                        if self._is_valid_square(binary_mask, top, left, size):
                            patches.append((size, top, left))
        
        return patches
    
    def _is_valid_square(self, binary_mask, top, left, size):
        """Verify that a square region is entirely within valid background"""
        H, W = binary_mask.shape
        
        # Check bounds
        if top < 0 or left < 0 or top + size > H or left + size > W:
            return False
            
        # Check if entire square is valid background
        square_region = binary_mask[top:top+size, left:left+size]
        return np.all(square_region == 1)
    
    def _select_non_overlapping_patches(self, patches, mask_shape):
        """Select non-overlapping patches from the sorted list"""
        H, W = mask_shape
        selected = []
        occupied_mask = np.zeros((H, W), dtype=bool)
        
        for size, top, left in patches:
            bottom = top + size
            right = left + size
            
            # Check bounds
            if bottom > H or right > W:
                continue
                
            # Check for overlap if non_overlapping is enabled
            if self.overlap:
                region = occupied_mask[top:bottom, left:right]
                if region.any():
                    continue
            
            # Mark region as occupied
            occupied_mask[top:bottom, left:right] = True
            selected.append((top, left, size))
            
            # Stop if we have enough patches
            if len(selected) >= self.k:
                break
        
        return selected


class CustomBackgroundCrop:
    def __init__(self, k=5, non_overlapping=True, target_size=336, min_patch_size=32):
        self.cropper = TopKBackgroundPatch(k=k, non_overlapping=non_overlapping, min_patch_size=min_patch_size)
        self.target_size = target_size
        self.resize_transform = transforms.Resize(
            target_size, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )

    def __call__(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """
        Apply background cropping to tensor inputs
        
        Args:
            image_tensor: Input image tensor of shape (C, H, W)
            mask_tensor: Mask tensor of shape (C, H, W) or (H, W)
            
        Returns:
            Resized cropped tensor
        """
        print(f"Input image shape: {image_tensor.shape}")
        print(f"Input mask shape: {mask_tensor.shape}")
        
        crops, crop_coords = self.cropper(image_tensor, mask_tensor)
        
        if len(crops) == 0:
            print('No valid crop found - falling back to center crop')
            # Apply center crop
            min_dim = min(image_tensor.shape[1], image_tensor.shape[2])
            center_crop = transforms.CenterCrop(min_dim)
            cropped = center_crop(image_tensor)
        else:
            print(f'Valid Crop Found! Crop coords: {crop_coords[0]}')
            cropped = crops[0]  # This will be the randomly selected patch
            
        # Resize the cropped tensor
        resized = self.resize_transform(cropped)
        print(f"Final output shape: {resized.shape}")
        return resized