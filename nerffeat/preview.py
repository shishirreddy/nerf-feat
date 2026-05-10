
from pathlib import Path

import cv2
import torch

from nerffeat.utils import embedding_to_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def save_rgb_preview(path: str | Path, image: torch.Tensor) -> None:
    cv2.imwrite(path, image.cpu().numpy() * 255)


def save_pose_encoder_previews(
    *,
    output_dir: str,
    preview_id: str,
    rgb_batch: torch.Tensor,
    mask_logits: torch.Tensor,
    image_feature_map: torch.Tensor,
) -> None:
    out = Path(output_dir)
    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)
    mask_preview = torch.sigmoid(mask_logits[0]).detach().cpu().numpy()
    cv2.imwrite(str(out / f"{preview_id}_mask.jpg"), (mask_preview / (0.01 + mask_preview.max())) * 255)
    cv2.imwrite(str(out / f"{preview_id}_feat.jpg"), embedding_to_rgb(image_feature_map[0].detach().cpu(), demean=True).cpu().numpy() * 255)
    cv2.imwrite(str(out / f"{preview_id}_target.jpg"), ((rgb_batch[0].cpu() * std) + mean).numpy() * 255)
