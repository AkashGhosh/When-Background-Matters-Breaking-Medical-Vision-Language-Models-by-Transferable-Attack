from surrogate.base import BaseAttentionExtractor
from surrogate.Blip import Blip2Extractor
from surrogate.Blipcoco import BlipCocoExtractor
from surrogate.ClipB16 import ClipB16Extractor
from surrogate.ClipB32 import ClipB32Extractor
from surrogate.ClipL336 import ClipL336Extractor
from surrogate.ClipLaion import ClipLaionExtractor

MODEL_CLASS_MAP = {
    "Salesforce/blip2-opt-2.7b": Blip2Extractor,
    "Salesforce/blip-itm-base-coco": BlipCocoExtractor,
    "openai/clip-vit-base-patch16": ClipB16Extractor,
    "openai/clip-vit-base-patch32": ClipB32Extractor,
    "openai/clip-vit-large-patch14-336": ClipL336Extractor,
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K": ClipLaionExtractor,
}