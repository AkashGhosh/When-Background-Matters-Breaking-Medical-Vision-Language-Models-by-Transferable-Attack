from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class BlackboxConfig:
    model_name: str = "gpt4v"
    batch_size: int = 1
    timeout: int = 30


@dataclass
class DataConfig:
    batch_size: int = 1
    num_samples: int = 5
    cle_data_path: str = "resources/images/bigscale"    # path for clean images
    tgt_data_path: str = "resources/images/target_images"  # path for target images
    mask_data_path: str = 'resources/images/masks'         # path for masks
    output: str = "./img_output"


@dataclass
class OptimConfig:
    alpha: float = 1.0
    epsilon: int = 4
    steps: int = 300


@dataclass
class ModelConfig:
    k: int = 10
    non_overlap: bool = True
    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"
    backbone: List[str] = field(default_factory=lambda: ["L336", "B16", "B32", "Laion"])


@dataclass
class MainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    attack: str = "fgsm"


@dataclass
class Ensemble3ModelsConfig(MainConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        use_source_crop=True,
        use_target_crop=True,
        backbone=["B16", "B32", "Laion"]
    ))
