# --- START OF FILE base.py ---

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BaseAttentionExtractor:
    def __init__(self, model):
        # Initialize hooks first to prevent AttributeError if subclass __init__ fails
        self.hooks = []
        self.attn_outputs = {} 
        self.model = model
        # The rest of the subclass __init__ will run after this, including register_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()