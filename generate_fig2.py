import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import the classes from the notebook code we extracted
try:
    from inference_utils import PIDNetFinetuner, PIDNetPreprocessor, yolo_to_mask, generate_tile_boxes
except ImportError:
    print("Error: inference_utils.py not found. Please ensure it is generated.")
    sys.exit(1)

def load_model(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    model = PIDNetFinetuner.load_from_checkpoint(
        ckpt_path,
        map_location='cpu',
        strict=False
    )
    model.eval()
    return model

def inference_naive(model, image_np):
    # Naive baseline resizes the entire image to 1024x1024 and does NOT use RGCr
    preprocessor = PIDNetPreprocessor(size=(1024, 1024), use_rgcr=False)
    encoded = preprocessor(image_np, np.zeros_like(image_np[:,:,0]))
    pixel_values = encoded['pixel_values'].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(pixel_values)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        # Upsample from 1/8 resolution back to 1024x1024
        logits = torch.nn.functional.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
            
    probs = torch.softmax(logits, dim=1)[0]
    pred_mask = probs.argmax(dim=0).numpy().astype(np.uint8)
    
    # Resize prediction back to native resolution for comparison
    h, w = image_np.shape[:2]
    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_mask

def inference_tiling(model, image_np):
    # Proposed model uses Tiling and RGCr color space
    h, w = image_np.shape[:2]
    preprocessor = PIDNetPreprocessor(size=(1024, 1024), use_rgcr=True)
    
    prob_map = torch.zeros((3, h, w), dtype=torch.float32)
    count_map = torch.zeros((h, w), dtype=torch.float32)
    
    # Test-time exhaustive tiling (overlap=0.0 as defined in your paper's config)
    boxes = generate_tile_boxes(h, w, (1024, 1024), 0.0)
    
    for (x0, y0, x1, y1) in boxes:
        tile = image_np[y0:y1, x0:x1]
        
        # Pad tile to 1024x1024 if it's an edge tile
        th, tw = tile.shape[:2]
        padded = np.zeros((1024, 1024, 3), dtype=np.uint8)
        padded[:th, :tw, :] = tile
        
        encoded = preprocessor(padded, np.zeros((1024, 1024)))
        pixel_values = encoded['pixel_values'].unsqueeze(0)
        
        with torch.no_grad():
            logits = model(pixel_values)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            # Upsample from 1/8 resolution back to 1024x1024
            logits = torch.nn.functional.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
                
        probs = torch.softmax(logits, dim=1)[0]
        
        # Crop probabilities back to original tile size to remove padding
        probs_cropped = probs[:, :th, :tw]
        
        prob_map[:, y0:y1, x0:x1] += probs_cropped
        count_map[y0:y1, x0:x1] += 1
        
    count_map[count_map == 0] = 1
    prob_map /= count_map
    pred_mask = prob_map.argmax(dim=0).numpy().astype(np.uint8)
    return pred_mask

def overlay_mask(image_np, mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # Fire = Bright Red, Smoke = Light Blue/Cyan
    color_mask[mask == 1] = [255, 0, 0]
    color_mask[mask == 2] = [0, 200, 255]
    
    alpha = 0.5
    overlay = image_np.copy()
    valid = mask > 0
    overlay[valid] = cv2.addWeighted(image_np[valid], 1-alpha, color_mask[valid], alpha, 0)
    return overlay

def main():
    img_path = "pruned_dataset_safe/images/20250107_EatonFire_wilson-s-mobo-c_70.jpg"
    lbl_path = "pruned_dataset_safe/labels/20250107_EatonFire_wilson-s-mobo-c_70.txt"
    
    base_ckpt = "checkpoints/pidnet-fire-smokeNEWDATASETRGB4real-epoch=48-val_fire_smoke_iou=0.708.ckpt"
    prop_ckpt = "checkpoints/pidnet-fire-smokeNEWDATASETRGCrAT4REAL2-epoch=45-val_fire_smoke_iou=0.756.ckpt"
    
    print("Loading image...")
    image_np = np.array(Image.open(img_path).convert('RGB'))
    
    print("Loading GT mask...")
    h, w = image_np.shape[:2]
    gt_mask = yolo_to_mask(lbl_path, h, w, 3)
    
    print("\n--- Baseline Model ---")
    baseline_model = load_model(base_ckpt)
    print("Running Naive RGB Inference (Resize to 1024x1024)...")
    base_mask = inference_naive(baseline_model, image_np)
    
    print("\n--- Proposed Model ---")
    proposed_model = load_model(prop_ckpt)
    print("Running Proposed Inference (RGCr + SAHI Tiling)...")
    prop_mask = inference_tiling(proposed_model, image_np)
    
    print("\nGenerating Overlays...")
    gt_overlay = overlay_mask(image_np, gt_mask)
    base_overlay = overlay_mask(image_np, base_mask)
    prop_overlay = overlay_mask(image_np, prop_mask)
    
    print("Plotting Figure 2...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Original
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=18, fontweight='bold')
    axes[0].axis('off')
    
    # 2. GT
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth", fontsize=18, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Baseline
    axes[2].imshow(base_overlay)
    axes[2].set_title("Baseline (Naive RGB)", fontsize=18, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Proposed
    axes[3].imshow(prop_overlay)
    axes[3].set_title("Proposed (RGCr + Tiling)", fontsize=18, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    out_path = "Figure_2_Qualitative_Comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Saved beautifully formatted figure to {out_path}")

if __name__ == "__main__":
    main()
