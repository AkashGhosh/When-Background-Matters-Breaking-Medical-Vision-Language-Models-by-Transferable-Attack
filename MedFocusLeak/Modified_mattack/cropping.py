import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
import random

class TopKBackgroundPatch:
    def __init__(self, k=3, non_overlapping=True):
        self.k = k
        self.overlap = non_overlapping
        
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
            
        # Create binary mask
        binary_mask = (mask_np < 128).astype(np.uint8)
        
        H, W = binary_mask.shape
        dp = np.zeros((H, W), dtype=int)
        patches = []
        
        for i in range(H):
            for j in range(W):
                if binary_mask[i][j] == 0:
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                    size = dp[i][j]
                    if size > 0:
                        top = i - size + 1
                        left = j - size + 1
                        patches.append((size, top, left))
                        
        patches.sort(reverse=True, key=lambda x: x[0])
        
        selected = []
        occupied_mask = np.zeros((H, W), dtype=bool)
        
        for size, top, left in patches:
            bottom = top + size
            right = left + size
            if bottom > H or right > W:
                continue
            region = occupied_mask[top:bottom, left:right]
            if self.overlap and region.any():
                continue
            occupied_mask[top:bottom, left:right] = True
            selected.append((top, left, size))
            if len(selected) >= self.k:
                break
        
        # Randomly select one patch from the top k patches
        if len(selected) > 0:
            chosen_patch = random.choice(selected)
            # Crop using tensor operations
            top, left, size = chosen_patch
            cropped_tensor = image_tensor[:, top:top+size, left:left+size]
            return [cropped_tensor], [chosen_patch]
        else:
            return [], []


class CustomBackgroundCrop:
    def __init__(self, k=1, non_overlapping=True, target_size=336):
        self.cropper = TopKBackgroundPatch(k=k, non_overlapping=non_overlapping)
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
        crops, crop_coords = self.cropper(image_tensor, mask_tensor)
        
        if len(crops) == 0:
            print('No valid crop found.')
            # Apply center crop
            min_dim = min(image_tensor.shape[1], image_tensor.shape[2])
            center_crop = transforms.CenterCrop(min_dim)
            cropped = center_crop(image_tensor)
        else:
            print('Valid Crop Found!')
            cropped = crops[0]  # This will be the randomly selected patch
            
        # Resize the cropped tensor
        resized = self.resize_transform(cropped)
        return resized