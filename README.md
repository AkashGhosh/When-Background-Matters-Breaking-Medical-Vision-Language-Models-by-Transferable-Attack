# When-Background-Matters-Breaking-Medical-Vision-Language-Models-by-Transferable-Attack
[ ACL 2026 ORAL ] MedFocusLeak is a transferable black-box multimodal adversarial attack that injects imperceptible perturbations into non-diagnostic background regions and shifts model attention to induce plausible but incorrect medical diagnoses.


When-Background-Matters-Breaking-Medical-Vision-Language-Models-by-Transferable-Attack/
└── MedFocusLeak/
    ├── DataProcessing/                          # Data Processing 
    │   ├── Adv_Text.py                          # Generating Adversarial Description from the real Medical Description
    │   └── White_Img.py                         # Generating White Images with the Adversarial Text written in it
    ├── Modified_mattack/
    │   ├── config/
    │   ├── resources/
    │   ├── surrogates/
    │   ├── .gitignore
    │   ├── LICENSE
    │   ├── README.md
    │   ├── blackbox_text_generation.py
    │   ├── config_schema.py
    │   ├── cropping.py
    │   ├── cropping2.py
    │   ├── evaluation_metrics.py
    │   ├── generate_adversarial_samples.py
    │   ├── generate_adversarial_samples2.py    # (Modified Code to add noise in the background of the images using the mask)
    │   ├── gpt_evaluate.py
    │   ├── keyword_matching_gpt.py
    │   ├── new.txt
    │   ├── requirements.txt
    │   └── utils.py
    ├── MultimodalFusion/
    │   └── Target_Img_gen.py                   # Multi-modal fusion for generation Target Images
    └── attentionshift/
        ├── Imgs/
        ├── masks/
        ├── surrogate/
        ├── Attack.py
        ├── Attack2.py
        ├── Visualize.py
        ├── Visualize2.py
        ├── new.txt
        ├── run_attack.py
        └── run_attack2.py                      # Final Step attention-shift
