import os
import torch
import torch.nn as nn
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------- Feature Extractor ---------------- #
class FeatureExtractor:
    """
    Extract intermediate features from a PyTorch model using hooks.
    """
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.features: Dict = {}
        self._handles = []

    def _get_hook(self, layer_name: str):
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.features[layer_name] = output[0]
            else:
                self.features[layer_name] = output
        return hook

    def attach(self):
        if self._handles:
            return
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handle = module.register_forward_hook(self._get_hook(name))
                self._handles.append(handle)
        if len(self._handles) != len(self.target_layers):
            raise ValueError("Could not find all target layers in the model.")

    def get_features(self) -> Dict:
        captured = self.features
        self.features = {}
        return captured

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# ---------------- BSA Loss ---------------- #
class BSALoss(nn.Module):
    """
    Block-wise Similarity Attack (BSA) loss.
    """
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, clean_features: Dict, adv_features: Dict) -> torch.Tensor:
        total_loss = 0.0
        assert clean_features.keys() == adv_features.keys()

        for layer_name in clean_features.keys():
            clean_f = clean_features[layer_name]
            adv_f = adv_features[layer_name]

            clean_f_flat = clean_f.view(-1, clean_f.size(-1))
            adv_f_flat = adv_f.view(-1, adv_f.size(-1))

            sim = self.cosine_similarity(clean_f_flat, adv_f_flat)
            total_loss += sim.sum()

        return total_loss


# ---------------- PGD-BSA Attack ---------------- #
def pgd_bsa_attack(
    model: nn.Module,
    processor,
    image,
    text: str,
    feature_extractor: FeatureExtractor,
    bsa_loss_fn: BSALoss,
    eps: float = 16/255,
    alpha: float = 2/255,
    steps: int = 40
):
    device = next(model.parameters()).device

    # Truncate text automatically if too long
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        truncation=True,        # <- ensures sequence length <= 77
        padding="max_length"
    ).to(device)
    clean_image_tensor = inputs['pixel_values']

    # Step 1: clean features
    with torch.no_grad():
        with feature_extractor as extractor:
            model(**inputs)
            clean_features = {k: v.detach() for k, v in extractor.get_features().items()}

    # Step 2: init adv image
    adv_image_tensor = clean_image_tensor.clone().detach()
    adv_image_tensor.requires_grad = True

    # Step 3: PGD iterations
    for _ in range(steps):
        if adv_image_tensor.grad is not None:
            adv_image_tensor.grad.zero_()

        adv_inputs = {"pixel_values": adv_image_tensor, "input_ids": inputs["input_ids"]}

        with feature_extractor as extractor:
            model(**adv_inputs)
            adv_features = extractor.get_features()

        loss = bsa_loss_fn(clean_features, adv_features)
        loss.backward()

        with torch.no_grad():
            adv_image_tensor.data = adv_image_tensor.data - alpha * adv_image_tensor.grad.sign()
            delta = torch.clamp(adv_image_tensor - clean_image_tensor, min=-eps, max=eps)
            adv_image_tensor.data = torch.clamp(clean_image_tensor + delta, min=0, max=1)

    return adv_image_tensor.detach(), clean_image_tensor.detach()


# ---------------- Attack from CSV ---------------- #
def attack_from_csv(
    csv_path: str,
    image_col: str,
    text_col: str,
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    eps: float = 16/255,
    alpha: float = 1/255,
    steps: int = 100,
    save_perturbation: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + processor
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Layers to extract
    vision_layers = [f"vision_model.encoder.layers.{i}" for i in range(model.config.vision_config.num_hidden_layers)]
    text_layers = [f"text_model.encoder.layers.{i}" for i in range(model.config.text_config.num_hidden_layers)]
    target_layers = vision_layers + text_layers

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    feature_extractor = FeatureExtractor(model, target_layers)
    bsa_loss_fn = BSALoss()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row[image_col]
        text = row[text_col]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[!] Could not open {img_path}: {e}")
            continue

        adv_image_tensor, clean_image_tensor = pgd_bsa_attack(
            model, processor, image, text,
            feature_extractor, bsa_loss_fn,
            eps=eps, alpha=alpha, steps=steps
        )

        adv_np = adv_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        clean_np = clean_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        adv_pil = Image.fromarray((adv_np * 255).astype(np.uint8))

        filename = os.path.basename(img_path)
        adv_path = os.path.join(output_dir, f"{filename}")
        adv_pil.save(adv_path)

        if save_perturbation:
            perturb = (adv_np - clean_np)
            perturb = (perturb - perturb.min()) / (perturb.max() - perturb.min() + 1e-8)
            perturb_pil = Image.fromarray((perturb * 255).astype(np.uint8))
            perturb_path = os.path.join(output_dir, f"perturb_{filename}")
            perturb_pil.save(perturb_path)

        print(f"Saved adversarial image to {adv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_col", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--eps", type=float, default=16/255)
    parser.add_argument("--alpha", type=float, default=1/255)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save_perturbation", action="store_true")

    args = parser.parse_args()

    attack_from_csv(
        csv_path=args.csv_path,
        image_col=args.image_col,
        text_col=args.text_col,
        output_dir=args.output_dir,
        model_name=args.model_name,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        save_perturbation=args.save_perturbation
    )
