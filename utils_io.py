import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from budget_sampler import uniform_sample


def load_image_as_tensor(path, device="cpu", img_size=512):
    img = Image.open(path).convert("RGB")
    if img_size is not None and int(img_size) > 0:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((int(img_size), int(img_size)), resampling)
    arr = np.asarray(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_uint8_image(t):
    t = t.detach().clamp(0, 1).cpu()
    _, c, _, _ = t.shape
    if c == 1:
        arr = t.squeeze(0).squeeze(0).numpy()
    else:
        arr = t.squeeze(0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype("uint8")


def tensor_to_pil(t):
    return Image.fromarray(tensor_to_uint8_image(t))


def save_tensor_to_png(t, path):
    imageio.imwrite(path, tensor_to_uint8_image(t))


def build_inpaint_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (gray < 30).astype(np.uint8)


def inpaint_rgb_image(rgb):
    mask = build_inpaint_mask(rgb)
    if not np.any(mask):
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def inpaint_pil_image(image):
    rgb = np.asarray(image.convert("RGB"))
    return Image.fromarray(inpaint_rgb_image(rgb))


def tensor_to_inpainted_pil(t):
    return inpaint_pil_image(tensor_to_pil(t))


def create_run_dir(log_dir, exp_name):
    os.makedirs(log_dir, exist_ok=True)
    count = [
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith(f"{exp_name}_")
    ]
    run_dir = os.path.join(log_dir, f"{exp_name}_{len(count) + 1:03d}")
    os.mkdir(run_dir)
    return run_dir


def initialize_run_outputs(args, image_tensor):
    run_dir = create_run_dir(args.log_dir, args.exp_name)
    uniform_path = os.path.join(run_dir, "uniform_sampled.png")
    llmind_path = os.path.join(run_dir, "llmind_sampled.png")
    results_path = os.path.join(run_dir, "results.json")

    uniform_sampled = uniform_sample(image_tensor, p=args.percentage)
    save_tensor_to_png(uniform_sampled, uniform_path)
    save_tensor_to_png(uniform_sampled, llmind_path)

    return run_dir, uniform_sampled, uniform_path, llmind_path, results_path


def interpolate_(img_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open {img_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filled_rgb = inpaint_rgb_image(rgb)
    save_name = os.path.join(save_folder, os.path.basename(img_path).replace(".png", "_filled.png"))
    Image.fromarray(filled_rgb).save(save_name)


def save_interpolated_pil(image, path):
    image.save(path)
    save_folder = os.path.dirname(path) or "."
    interpolate_(path, save_folder)
    filled_path = os.path.join(save_folder, os.path.basename(path).replace(".png", "_filled.png"))
    os.replace(filled_path, path)
