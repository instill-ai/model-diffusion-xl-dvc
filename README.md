---
Task: TextToImage
Tags:
  - TextToImage
  - Text-To-Image
  - Diffusion-xl
---

# Model-Diffusion-XL-DVC

🔥🔥🔥 Deploy state-of-the-art [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) model in PyTorch format via open-source [VDP](https://github.com/instill-ai/vdp).

- This repository contains the base of refiner model of stable diffution xl
- Disk Space Requirements: 38G
- GPU Memory Requirements: 15G

**Create Model**

```json
{
    "id": "stable-diffusion-xl-gpu",
    "description": "Stable-Diffusion-XL, from StabilityAI, is trained to generate image based on your prompts.",
    "model_definition": "model-definitions/container",
    "visibility": "VISIBILITY_PUBLIC",
    "region": "REGION_GCP_EUROPE_WEST_4",
    "hardware": "GPU",
    "configuration": {
        "task": "TASK_TEXT_TO_IMAGE"
    }
}
```

**Inference model**

```json
{
    "task_inputs": [
        {
            "text_to_image": {
                "prompt": "Mona lisa",
                "steps": "50",
                "cfg_scale": "5.5",
                "seed": "1",
                "samples": 1
            }
        }
    ]
}
```
