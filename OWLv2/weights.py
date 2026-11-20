import torch
from torch import nn
from transformers import Owlv2Config, Owlv2ForObjectDetection

from detector import Owlv2Detector


def get_new_key(checkpoint_key: str) -> str | None:
    if checkpoint_key.startswith("owlv2.vision_model."):
        rest = checkpoint_key[len("owlv2.vision_model.") :]
        rest = rest.replace("pre_layernorm.", "pre_layrnorm.")
        return "clip.vision." + rest
    if checkpoint_key.startswith("owlv2.text_model."):
        return "clip.text." + checkpoint_key[len("owlv2.text_model.") :]
    if checkpoint_key.startswith("owlv2.text_projection."):
        return "clip." + checkpoint_key[len("owlv2.") :]
    if checkpoint_key.startswith("owlv2.visual_projection."):
        return "clip." + checkpoint_key[len("owlv2.") :]
    if checkpoint_key == "owlv2.logit_scale":
        return "clip.logit_scale"
    if checkpoint_key.startswith(("class_head.", "box_head.", "objectness_head.")):
        return checkpoint_key
    if checkpoint_key.startswith("layer_norm."):
        return "patch_merger." + checkpoint_key
    return None


def load_detector(
    model_name: str = "google/owlv2-base-patch16-ensemble",
    device: str | torch.device = "cpu",
) -> tuple[Owlv2Detector, Owlv2Config]:
    config = Owlv2Config.from_pretrained(model_name)
    reference = Owlv2ForObjectDetection.from_pretrained(model_name, attn_implementation="eager")
    detector = Owlv2Detector(config)

    mapped = {}
    skipped = []
    for key, value in reference.state_dict().items():
        new_key = get_new_key(key)
        if new_key is None:
            skipped.append(key)
            continue
        mapped[new_key] = value

    missing, unexpected = detector.load_state_dict(mapped, strict=False)
    real_missing = [key for key in missing if "position_ids" not in key and key != "box_bias"]
    if real_missing or unexpected or skipped:
        raise RuntimeError("Model was not ported correctly!")
    detector.to(device)
    detector.eval()
    del reference
    return detector, config


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
