import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from surrogate.Blip import Blip2Extractor


class AttentionPerturber:
    def __init__(self, extractor:Blip2Extractor, image:Image.Image, mask:Image.Image, attention_layers:dict, target_layers=None):
        self.extractor = extractor
        self.model = extractor.model
        self.device = self.model.device
        self.original_size = image.size
        self.target_layers = target_layers
        
        target_size = (
            self.extractor.processor.image_processor.size['height'],
            self.extractor.processor.image_processor.size['width']
        )
        
        # Corrected typo from 'premute' to 'permute'
        image_resized = image.resize(target_size, Image.Resampling.BICUBIC)
        image_np = np.array(image_resized) / 255.0
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)
        
        self.perturbed_tensor = image_tensor.to(self.device, torch.float16).clone()
        self.original_tensor = image_tensor.to(self.device, torch.float16).clone()
        
        self.image_mean = torch.tensor(self.extractor.processor.image_processor.image_mean).view(1, 3, 1, 1).to(self.device, torch.float16)
        self.image_std = torch.tensor(self.extractor.processor.image_processor.image_std).view(1, 3, 1, 1).to(self.device, torch.float16)
        
        patch_size = self.model.config.vision_config.patch_size
        self.grid_size = target_size[0] // patch_size
        
        mask_resized = mask.convert("L").resize((self.grid_size, self.grid_size), Image.Resampling.NEAREST)
        mask_np = np.array(mask_resized) / 255.0
        self.foreground_mask = (torch.from_numpy(mask_np).to(self.device, torch.float16) > 0.5).float()
        self.background_mask = 1.0 - self.foreground_mask
        
        self.fg_sum = self.foreground_mask.sum() + 1e-8
        self.bg_sum = self.background_mask.sum() + 1e-8
        
        # This is for reference only; the loss function re-computes attention
        self.original_attention_grids = attention_layers
        
    def calculate_loss(self, image_to_process):
        """
        --- THIS IS THE MOST CRITICAL CORRECTION ---
        This function now correctly computes the loss while keeping the computation graph intact.
        It does NOT call `self.extractor.forward()`.
        """
        # Normalize the tensor for the model input
        norm_img_tensor = (image_to_process - self.image_mean) / self.image_std
        
        # Clear any old attention maps from previous runs
        self.extractor.attn_outputs.clear()
        
        # Directly call the vision model to run a forward pass.
        # This ensures gradients are tracked from the input `norm_img_tensor`.
        _ = self.model.vision_model(pixel_values=norm_img_tensor, output_attentions=True)
        
        # The hooks in the extractor have now populated `attn_outputs` with tensors
        # that are still part of the current computation graph.
        img_atten = self.extractor.attn_outputs
        
        total_loss = 0
        layers_to_attack = self.target_layers if self.target_layers is not None else img_atten.keys()
        
        for layer in layers_to_attack:
            if layer not in img_atten:
                continue
                
            atten_map = img_atten[layer] # This tensor is on the GPU and has grad_fn
            cls_atten = atten_map[0, :, 0, 1:].mean(dim=0) # Mean over attention heads
            atten_grid = cls_atten.reshape(self.grid_size, self.grid_size)
            
            mean_fg_attention = (atten_grid * self.foreground_mask).sum() / self.fg_sum
            mean_bg_attention = (atten_grid * self.background_mask).sum() / self.bg_sum
            
            # The goal is to minimize foreground attention and maximize background attention.
            # Minimizing (FG - BG) achieves this.
            layer_loss = mean_fg_attention - mean_bg_attention
            total_loss += layer_loss
            
        return total_loss
    
    def perturb(self, num_steps=50, alpha=4/255, epsilon=4/255):
        print("Running PGD optimization...")
        for i in tqdm(range(num_steps), desc="PGD Steps"):
            self.perturbed_tensor.requires_grad = True
            
            # Zero out any previous gradients on the model parameters
            self.model.zero_grad()
            
            loss = self.calculate_loss(self.perturbed_tensor)
            
            if torch.isnan(loss):
                print("Warning: Loss is NaN, stopping perturbation.")
                break
            
            loss.backward()
            
            grad = self.perturbed_tensor.grad
            
            # Check if gradient calculation was successful
            if grad is None:
                print("\n\n--- FATAL ERROR ---")
                print("Gradient is None. The computational graph is broken.")
                print("This is likely because the `calculate_loss` function in Attack.py or the hook in Blip.py is incorrect.")
                print("Please ensure you have replaced BOTH files with the provided corrected versions.")
                print("---------------------\n\n")
                # Return the original image as a failure signal
                return self.tensor_to_pil(self.original_tensor, target_size=self.original_size)

            with torch.no_grad():
                # PGD update step
                new_tensor = self.perturbed_tensor - alpha * grad.sign()
                perturbation = torch.clamp(new_tensor - self.original_tensor, -epsilon, epsilon)
                self.perturbed_tensor = torch.clamp(self.original_tensor + perturbation, 0, 1)
                
            if (i+1) % 10 == 0 or i == num_steps-1:
                print(f"Step {i+1}/{num_steps}, Loss: {loss.item():.4f}, Grad Norm: {grad.norm().item():.4f}")
            
        print("Perturbation finished.")
        return self.tensor_to_pil(self.perturbed_tensor, target_size=self.original_size)
    
    def tensor_to_pil(self, tensor, target_size=None):
        """Converts a tensor to a PIL image, with optional resizing."""
        image_np = (tensor.detach().squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        if target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.BICUBIC)
        return pil_image