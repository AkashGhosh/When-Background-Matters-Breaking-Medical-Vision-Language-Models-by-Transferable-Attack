# --- START OF FILE Attack.py ---
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from surrogate.base import BaseAttentionExtractor


class AttentionPerturber:
    # --- KEY CHANGE: __init__ now accepts a list of pre-loaded extractors ---
    def __init__(self, extractors: List[BaseAttentionExtractor], image: Image.Image, mask: Image.Image):
        self.device = extractors[0].model.device
        self.original_size = image.size
        self.loaded_model_configs = []
        
        self.perturbed_tensor = None
        self.original_tensor = None

        # --- KEY CHANGE: Use the provided extractors directly, no model loading here ---
        for i, extractor in enumerate(extractors):
            # The config was attached to the extractor in run_attack.py
            config = extractor.manual_config
            
            target_size = config["input_size"]
            patch_size = config["patch_size"]
            image_mean = torch.tensor(config["image_mean"]).view(1, 3, 1, 1).to(self.device, torch.float16)
            image_std = torch.tensor(config["image_std"]).view(1, 3, 1, 1).to(self.device, torch.float16)
            
            if i == 0:
                image_resized = image.resize(target_size, Image.Resampling.BICUBIC)
                image_np_resized = np.array(image_resized) / 255.0
                image_tensor_resized = torch.tensor(image_np_resized).permute(2, 0, 1).unsqueeze(0)
                
                self.perturbed_tensor = image_tensor_resized.to(self.device, torch.float16).clone()
                self.original_tensor = image_tensor_resized.to(self.device, torch.float16).clone()

            grid_size = target_size[0] // patch_size
            
            mask_resized = mask.convert("L").resize((grid_size, grid_size), Image.Resampling.NEAREST)
            mask_np = np.array(mask_resized) / 255.0
            foreground_mask = (torch.from_numpy(mask_np).to(self.device, torch.float16) > 0.5).float()
            
            prepared_config = {
                "extractor": extractor, "target_size": target_size, "image_mean": image_mean, "image_std": image_std,
                "grid_size": grid_size, "foreground_mask": foreground_mask, "background_mask": 1.0 - foreground_mask,
                "fg_sum": foreground_mask.sum() + 1e-8, "bg_sum": (1.0 - foreground_mask).sum() + 1e-8,
            }
            self.loaded_model_configs.append(prepared_config)

    def calculate_ensemble_loss(self, image_to_process):
        total_ensemble_loss = 0
        
        for config in self.loaded_model_configs:
            extractor = config['extractor']
            model = extractor.model
            
            resized_image_tensor = F.interpolate(
                image_to_process, size=config['target_size'], mode='bicubic', align_corners=False
            )
            
            norm_img_tensor = (resized_image_tensor - config['image_mean']) / config['image_std']
            
            extractor.attn_outputs.clear()
            _ = model.vision_model(pixel_values=norm_img_tensor, output_attentions=True)
            img_atten = extractor.attn_outputs
            
            model_loss = 0
            if not img_atten: continue

            for layer in img_atten.keys():
                atten_map = img_atten[layer]
                cls_atten = atten_map[0, :, 0, 1:].mean(dim=0)
                atten_grid = cls_atten.reshape(config['grid_size'], config['grid_size'])
                
                mean_fg_attention = (atten_grid * config['foreground_mask']).sum() / config['fg_sum']
                mean_bg_attention = (atten_grid * config['background_mask']).sum() / config['bg_sum']
                
                layer_loss = mean_fg_attention - mean_bg_attention
                model_loss += layer_loss
            
            total_ensemble_loss += model_loss
            
        return total_ensemble_loss
    
    def perturb(self, num_steps=50, alpha=4/255, epsilon=4/255):
        print("\nRunning PGD optimization on the model ensemble...")
        for i in tqdm(range(num_steps), desc="PGD Steps"):
            self.perturbed_tensor.requires_grad = True
            
            for config in self.loaded_model_configs:
                config['extractor'].model.zero_grad()
            
            loss = self.calculate_ensemble_loss(self.perturbed_tensor)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()}, stopping perturbation.")
                break
            
            loss.backward()
            grad = self.perturbed_tensor.grad
            
            if grad is None:
                print("FATAL ERROR: Gradient is None. The computational graph is broken.")
                return self.tensor_to_pil(self.original_tensor, target_size=self.original_size)

            with torch.no_grad():
                new_tensor = self.perturbed_tensor - alpha * grad.sign()
                perturbation = torch.clamp(new_tensor - self.original_tensor, -epsilon, epsilon)
                self.perturbed_tensor = torch.clamp(self.original_tensor + perturbation, 0, 1)
                
            if (i+1) % 10 == 0 or i == num_steps-1:
                print(f"Step {i+1}/{num_steps}, Ensemble Loss: {loss.item():.4f}, Grad Norm: {grad.norm().item():.4f}")
            
        print("Perturbation finished.")
        return self.tensor_to_pil(self.perturbed_tensor, target_size=self.original_size)
    
    def tensor_to_pil(self, tensor, target_size=None):
        image_np = (tensor.detach().squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        if target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.BICUBIC)
        return pil_image