import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from surrogate.base import BaseAttentionExtractor
from PIL import Image
import requests

class Blip2Extractor(BaseAttentionExtractor):
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        print(f"Loading model: {model_name}")
        # The 'attn_implementation="eager"' argument suppresses the warning you saw.
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
        )
        model.eval().cuda()
        super().__init__(model)
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.vision_config = self.model.vision_model.config
        self.register_hooks()

    def register_hooks(self):
        """
        Registers forward hooks on the self-attention modules of the vision encoder.
        """
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # --- THIS IS THE CRITICAL FIX ---
                # We store the attention tensor directly. 
                # It remains on the GPU and connected to the computation graph.
                # Do NOT use .detach() or .cpu() here.
                self.attn_outputs[layer_idx] = output[1]
            return hook_fn

        # The vision model is typically at self.model.vision_model
        for idx, layer in enumerate(self.model.vision_model.encoder.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def forward(self, image):
        """
        Performs a forward pass to get attention maps for a clean image.
        This function is intended for inference and does not track gradients.
        """
        self.attn_outputs.clear()
        # The processor prepares the image for the model
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device, torch.float16)
        pixel_values = inputs.pixel_values
        
        # We don't need gradients when just getting the original maps
        with torch.no_grad():
            # Pass through the vision model to trigger the hooks
            _ = self.model.vision_model(pixel_values=pixel_values, output_attentions=True)

        # The hooks have populated self.attn_outputs. We now detach and move to CPU
        # for external use, as the graph is no longer needed.
        detached_outputs = {k: v.detach().cpu() for k, v in self.attn_outputs.items()}
        return detached_outputs