#!/usr/bin/env python3
"""Build the perceptual ground truth: SSIM and LPIPS between generated images.

For each reference LoRA and each of its tampered copies, the same prompts and
seeds are used to sample images from a frozen SD 1.5 base model.  LPIPS between
the reference and the tampered images quantifies how much of the model's
behaviour survived the modification, and is what ``attack_breakdown.py`` uses as
ground truth:

* ``LPIPS <= 0.30``  -- the copy still behaves like the original (a plagiarism case);
* ``LPIPS >  0.30``  -- the modification changed the model's behaviour.

Results are written as ``<output-dir>/<lora-name>_metric_results.json``.

Example
-------
    export HF_TOKEN=hf_xxx        # only needed for gated base models
    python scripts/measure_lpips_ssim.py \
        --lora-name animal_SD1.5_cat_20230627113759 \
        --lora-files-root /data/hx/lora \
        --output-dir results/lpips_ssim_similarity
"""

import argparse
import gc
import glob
import json
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Diffusers & metrics
from diffusers import StableDiffusionPipeline
import lpips
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from huggingface_hub import login

warnings.filterwarnings("ignore", category=UserWarning)

# Only authenticate when a token is actually provided (gated base models).
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"))
from lora_metadata import LORA_METADATA, PROMPT_TEMPLATES, LORA_TYPES  # noqa: E402

# ==============================================================================
# 0. Global configuration
# ==============================================================================
BASE_MODEL_PATH = "Lykon/dreamshaper-8"

SEEDS = [42, 1234]
IMAGE_OUTPUT_ROOT = "lpips_ssim_comparison_images"
JSON_RESULTS_DIR = "results/lpips_ssim_similarity"
LORA_FILES_ROOT = "/data/hx/lora"
BENIGN_SUBDIR = "benign"
TAMPER_SUBFOLDERS = ['distill', 'extract', 'flip', 'noise', 'prune', 'ablation']

# ==============================================================================
# 1. Image generation
# ==============================================================================
def generate_image_with_lora(pipe, lora_path, prompt, seed, output_path, lora_weight):
    """
    Load LoRA -> generate image -> unload LoRA.
    """
    if os.path.exists(output_path):
        return

    try:
        # load the LoRA
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))
        
        # deterministic sampling
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # inference
        with torch.inference_mode():
            image = pipe(
                prompt, 
                num_inference_steps=30, 
                generator=generator, 
                cross_attention_kwargs={"scale": lora_weight}
            ).images[0]
            
        image.save(output_path)
        
    except Exception as e:
        print(f"[Error] generation failed ({os.path.basename(lora_path)}): {e}")
    finally:
        # always unload so the next iteration starts clean
        pipe.unload_lora_weights()

def run_generation_phase(args, prompts, original_path, tampered_paths, image_output_dir):
    print("\n" + "="*80)
    print("--- Phase 1: image generation (SD 1.5) ---")
    print("="*80)
    
    base_name = args.lora_name
    
    print(f"[*] loading SD 1.5 base model: {BASE_MODEL_PATH}")
    # SD 1.5 is small enough to fit on a single GPU
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.float16, 
        safety_checker=None, 
        requires_safety_checker=False, 
        use_safetensors=True, 
        variant="fp16"
    ).to("cuda:0")

    lora_metadata = LORA_METADATA.get(base_name, {})
    lora_weight = lora_metadata.get('recommended_weight', 1.0)

    # for each tampered LoRA: produce reference and tampered images
    for tampered_path in tqdm(tampered_paths, desc="tampered LoRAs"):
        # 1. reference images
        for prompt_idx, prompt in enumerate(prompts):
            for seed in SEEDS:
                original_img_name = f"original_{base_name}_seed{seed}_prompt{prompt_idx}.png"
                original_img_path = os.path.join(image_output_dir, original_img_name)
                generate_image_with_lora(pipe, original_path, prompt, seed, original_img_path, lora_weight)

        # 2. tampered images
        for prompt_idx, prompt in enumerate(prompts):
            for seed in SEEDS:
                tampered_img_name = f"tampered_{os.path.basename(tampered_path)}_seed{seed}_prompt{prompt_idx}.png"
                tampered_img_path = os.path.join(image_output_dir, tampered_img_name)
                generate_image_with_lora(pipe, tampered_path, prompt, seed, tampered_img_path, lora_weight)

    print("[*] image generation finished.")
    return pipe

# ==============================================================================
# 2. Image preprocessing (for the metrics)
# ==============================================================================
def load_image_for_metrics(path):
    return Image.open(path).convert("RGB")

def preprocess_for_lpips(pil_img, device):
    """[0, 255] PIL -> [-1, 1] Tensor"""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform(pil_img).unsqueeze(0).to(device)

def preprocess_for_ssim(pil_img):
    """PIL -> Numpy Array"""
    return np.array(pil_img)

# ==============================================================================
# 3. Metric computation (SSIM & LPIPS)
# ==============================================================================
def run_calculation_phase(base_name, prompts, original_path, tampered_paths, image_output_dir):
    print("\n" + "="*80)
    print("--- Phase 2: SSIM & LPIPS ---")
    print("="*80)
    
    calc_device = "cuda:0"
    
    print(f"[*] loading LPIPS (net='vgg') on {calc_device}...")
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(calc_device)
    lpips_loss_fn.eval()

    results = []
    
    for tampered_path in tqdm(tampered_paths, desc="similarity metrics"):
        ssim_scores = []
        lpips_scores = []
        
        for prompt_idx, prompt in enumerate(prompts):
            for seed in SEEDS:
                original_img_name = f"original_{base_name}_seed{seed}_prompt{prompt_idx}.png"
                original_img_path = os.path.join(image_output_dir, original_img_name)
                
                tampered_img_name = f"tampered_{os.path.basename(tampered_path)}_seed{seed}_prompt{prompt_idx}.png"
                tampered_img_path = os.path.join(image_output_dir, tampered_img_name)
                
                if os.path.exists(original_img_path) and os.path.exists(tampered_img_path):
                    img1_pil = load_image_for_metrics(original_img_path)
                    img2_pil = load_image_for_metrics(tampered_img_path)
                    
                    # --- SSIM ---
                    img1_np = preprocess_for_ssim(img1_pil)
                    img2_np = preprocess_for_ssim(img2_pil)
                    
                    # size alignment
                    if img1_np.shape != img2_np.shape:
                        img2_pil = img2_pil.resize(img1_pil.size)
                        img2_np = preprocess_for_ssim(img2_pil)

                    current_ssim = ssim(img1_np, img2_np, channel_axis=2, data_range=255)
                    ssim_scores.append(current_ssim)
                    
                    # --- LPIPS ---
                    img1_tensor = preprocess_for_lpips(img1_pil, calc_device)
                    img2_tensor = preprocess_for_lpips(img2_pil, calc_device)
                    
                    with torch.no_grad():
                        current_lpips = lpips_loss_fn(img1_tensor, img2_tensor).item()
                    lpips_scores.append(current_lpips)

        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
        avg_lpips = np.mean(lpips_scores) if lpips_scores else 0.0 
        
        results.append({
            "tampered_file": os.path.basename(tampered_path),
            "avg_ssim": round(avg_ssim, 4),     # 1.0 = identical
            "avg_lpips": round(avg_lpips, 4)    # 0.0 = identical
        })
    
    return results

# ==============================================================================
# Helpers
# ==============================================================================
def find_tampered_versions(base_name):
    found_files = []
    for folder in TAMPER_SUBFOLDERS:
        search_path = os.path.join(LORA_FILES_ROOT, folder, f"{base_name}*.safetensors")
        found_files.extend(glob.glob(search_path))
    return found_files

def get_prompts_for_lora(base_name):
    metadata = LORA_METADATA.get(base_name)
    if not metadata:
        raise ValueError(f"no entry for '{base_name}' in configs/lora_metadata.py")
    trigger_words = metadata.get("trigger_words", "")
    lora_type = LORA_TYPES.get(base_name, "character")
    template = PROMPT_TEMPLATES.get(lora_type)
    return [p.format(trigger_words=trigger_words) for p in template]

# ==============================================================================
# Main
# ==============================================================================
def main(args):
    global LORA_FILES_ROOT, JSON_RESULTS_DIR, IMAGE_OUTPUT_ROOT
    LORA_FILES_ROOT = args.lora_files_root
    JSON_RESULTS_DIR = args.output_dir
    IMAGE_OUTPUT_ROOT = args.image_dir

    os.makedirs(JSON_RESULTS_DIR, exist_ok=True)
    base_name = args.lora_name

    try:
        original_path = os.path.join(
            LORA_FILES_ROOT, args.benign_subdir, f"{base_name}.safetensors"
        )
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"reference LoRA not found: {original_path}")

        tampered_paths = find_tampered_versions(base_name)
        if not tampered_paths:
            print(f"no tampered versions found for '{base_name}'.")
            return

        prompts = get_prompts_for_lora(base_name)
        image_output_dir = os.path.join(IMAGE_OUTPUT_ROOT, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # --- Phase 1: image generation ---
        pipe = run_generation_phase(args, prompts, original_path, tampered_paths, image_output_dir)

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        # --- Phase 2: SSIM / LPIPS ---
        results = run_calculation_phase(
            base_name, prompts, original_path, tampered_paths, image_output_dir
        )

        print(f"\n--- {base_name} (SD 1.5) ---")
        results.sort(key=lambda x: x["tampered_file"])

        print(f"{'file':<50} | {'SSIM (higher)':<13} | {'LPIPS (lower)':<13}")
        print("-" * 84)
        for res in results:
            print(f"{res['tampered_file']:<50} | {res['avg_ssim']:.4f}        | {res['avg_lpips']:.4f}")

        output_json_path = os.path.join(JSON_RESULTS_DIR, f"{base_name}_metric_results.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n[*] results written to: {output_json_path}")

    except Exception as e:
        print(f"\n[error] experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perceptual ground truth (SSIM & LPIPS) for SD 1.5 LoRAs."
    )
    parser.add_argument("--lora-name", type=str, default="animal_SD1.5_cat_20230627113759",
                        help="reference LoRA name (without extension)")
    parser.add_argument("--lora-files-root", type=str, default=LORA_FILES_ROOT,
                        help="root holding the reference and tampered LoRA folders")
    parser.add_argument("--benign-subdir", type=str, default=BENIGN_SUBDIR,
                        help="subfolder of --lora-files-root holding the reference LoRAs")
    parser.add_argument("--output-dir", type=str, default=JSON_RESULTS_DIR,
                        help="where to write <lora-name>_metric_results.json")
    parser.add_argument("--image-dir", type=str, default=IMAGE_OUTPUT_ROOT,
                        help="where to cache the generated images")
    main(parser.parse_args())
