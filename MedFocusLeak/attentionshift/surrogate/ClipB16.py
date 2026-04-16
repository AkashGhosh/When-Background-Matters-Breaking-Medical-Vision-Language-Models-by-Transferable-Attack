import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from surrogate.base import BaseAttentionExtractor
from PIL import Image
import requests



class ClipB16Extractor(BaseAttentionExtractor):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        print(f"Loading model: {model_name}")
        model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        model.eval().cuda()
        super().__init__(model)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_config = self.model.vision_model.config
        self.register_hooks()

    def register_hooks(self):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                self.attn_outputs[layer_idx] = output[1]
            return hook_fn

        for idx, layer in enumerate(self.model.vision_model.encoder.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def forward(self, image):
        """
        Performs a forward pass to get attention maps for a clean image.
        This function is intended for inference and does not track gradients.
        """
        self.attn_outputs.clear()
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Step 2: Extract the 'pixel_values' tensor from the dictionary-like object.
        # Step 3: Now, move *this specific tensor* to the correct device and cast its dtype.
        pixel_values = inputs.pixel_values.to(self.model.device, torch.float16)

        # We don't need gradients when just getting the original maps
        with torch.no_grad():
            # Pass the corrected tensor through the vision model to trigger the hooks
            _ = self.model.vision_model(pixel_values=pixel_values, output_attentions=True)

        # The rest of the function is correct.
        detached_outputs = {k: v.detach().cpu() for k, v in self.attn_outputs.items()}
        return detached_outputs