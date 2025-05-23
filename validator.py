from PIL import Image
from typing import List
from validation.engine.validation_engine import ValidationEngine
from rmbg_birefnet import BackgroundRemover
import torch
import numpy as np

def convert_rgba_to_rgb_tensor(
    img: Image.Image,
    fill_pct: float = 0.85,
    out_size: tuple = None,
    bg_color: tuple = (255, 255, 255),
    resample=Image.LANCZOS
) -> torch.Tensor:
    if img.mode != 'RGBA':
        raise ValueError("Input image must be RGBA.")
    if not (0 < fill_pct <= 1):
        raise ValueError("fill_pct must be in (0, 1].")

    W, H = out_size if out_size else img.size
    bbox = img.getbbox()
    if not bbox:
        return torch.from_numpy(np.full((H, W, 3), bg_color, np.uint8))

    l, t, r, b = bbox
    ow, oh = r - l, b - t
    if ow <= 0 or oh <= 0:
        return torch.from_numpy(np.full((H, W, 3), bg_color, np.uint8))

    max_canvas = max(W, H)
    max_obj = max(ow, oh)
    scale = max_canvas * fill_pct / max_obj
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))

    img_resized = img.resize((nw, nh), resample)
    bg = Image.new('RGB', (W, H), bg_color)
    px, py = (W - nw) // 2, (H - nh) // 2
    bg.paste(img_resized, (px, py), img_resized.split()[3])

    return torch.from_numpy(np.array(bg, np.uint8))

validation_engine = ValidationEngine()
rmbg = BackgroundRemover()

class ImageValidator:
    def __init__(self, device='cuda'):
        self.validation_engine = validation_engine
        self.rembg_model = rmbg
        self.device = device
        print("ImageValidator initialized.")

    async def validate_and_select_top_k(
        self,
        images: List[Image.Image],
        refined_prompt: str,
        original_prompt: str,
        top_k: int
    ) -> List[Image.Image]:
        print(f"Validator received {len(images)} images for prompt: '{refined_prompt}'. Selecting top {top_k}.")

        # --- 1. Background Removal ---
        with torch.inference_mode():
            rgba_images = [self.rembg_model(img.resize((512, 512))) for img in images]
        valid_images = [img for img in rgba_images if img is not None]
        if not valid_images:
            raise RuntimeError("Background removal failed for all images.")

        # --- 2. Convert to tensor ---
        imgs_ = [
            convert_rgba_to_rgb_tensor(img, out_size=(224, 224)).to(self.device)
            for img in valid_images
        ]

        # --- 3. Tính điểm ---
        # Chú ý: bạn có thể cần đổi lại prompt cho đúng tên biến mà validation_engine yêu cầu
        clip_scores = self.validation_engine._text_vs_image_metric.score_text_alignment_2d(
            imgs_, original_prompt
        )
        quality_scores = self.validation_engine._image_quality_metric.score_images_quality_2d(imgs_)

        # --- 4. Gộp điểm, chọn top k ---
        scored_imgs = []
        for i, (img, clip_score, quality_score) in enumerate(zip(valid_images, clip_scores, quality_scores)):
            merge_score = 0.7 * quality_score + 0.3 * clip_score
            scored_imgs.append((merge_score, img))

        # Sắp xếp giảm dần theo merge_score, chọn top_k
        scored_imgs.sort(key=lambda x: x[0], reverse=True)
        selected_images = [img for _, img in scored_imgs[:top_k]]

        print(f"Validator selected {len(selected_images)} images.")
        return selected_images
