#!/usr/bin/env python3
"""
PIE-Bench Processing Script for SAM 3 + SD Inpainting

Processes PIE-Bench dataset with SAM 3 text-based segmentation + SD Inpainting.

PIE-Bench structure:
    data/
    ├── annotation_images/
    │   ├── 0_random_140/
    │   ├── 1_change_object_80/
    │   ├── 2_add_object_80/
    │   ├── ...
    └── mapping_file.json

Usage:
    python process_piebench.py --data_dir ./data --output_dir ./output
"""
from transformers import Sam3Processor, Sam3Model
import argparse
from utils.utils import txt_draw
import json
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from typing import Dict, List, Optional
import time
from tqdm import tqdm
import pycocotools.mask as mask_utils


# def load_sam3_model(checkpoint_path: str, device: str = "cuda"):
#     """Load SAM 3 model"""
#     from sam3 import build_sam3
#     from sam3.predictor import SAM3Predictor
    
#     print(f"Loading SAM 3 from {checkpoint_path}...")
#     model = build_sam3(checkpoint=checkpoint_path)
#     model = model.to(device)
#     model.eval()
    
#     return SAM3Predictor(model)

def load_sam3_model(model_id: str = "facebook/sam3", device: str = "cuda"):
    """
    Load SAM3 image model + processor from Hugging Face.

    Args:
        model_id: Hugging Face model ID or local path (e.g. "facebook/sam3").
        device: "cuda" or "cpu".

    Returns:
        (model, processor, device_str)
    """
    # Normalize device: if user asks for cuda but it's not available, fall back to cpu
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, using CPU for SAM3.")
        device = "cpu"

    print(f"Loading SAM3 from {model_id} on {device}...")
    model = Sam3Model.from_pretrained(model_id).to(device)
    processor = Sam3Processor.from_pretrained(model_id)

    model.eval()
    return model, processor, device


# switch this out for LaMa
def load_inpainting_model(model_id: str = "runwayml/stable-diffusion-inpainting", device: str = "cuda"):
    """Load Stable Diffusion Inpainting pipeline"""
    from diffusers import StableDiffusionInpaintPipeline
    
    print(f"Loading SD Inpainting: {model_id}...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    
    # Enable optimizations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    return pipe


def load_piebench_annotations(json_path: str) -> Dict:
    """Load PIE-Bench mapping file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def decode_rle_mask(rle_mask: Dict) -> np.ndarray:
    """Decode RLE mask from PIE-Bench format"""
    return mask_utils.decode(rle_mask).astype(bool)


def extract_edit_object(editing_instruction: str, blended_word: str) -> str:
    """
    Extract the object to segment from PIE-Bench annotation
    
    Args:
        editing_instruction: e.g., "Make the frame of the bike rusty"
        blended_word: e.g., "bicycle bicycle"
    
    Returns:
        Object name to segment (e.g., "bicycle")
    """
    # Blended word format is "source_word target_word"
    # We want to segment the source object
    words = blended_word.split()
    if len(words) >= 1:
        return words[0]  # Return first word (source object)
    
    # Fallback: try to extract from instruction
    # Common patterns: "the X", "X's", etc.
    import re
    match = re.search(r'the (\w+)', editing_instruction.lower())
    if match:
        return match.group(1)
    
    # Last resort: return blended_word as-is
    return blended_word


# def segment_with_sam3(sam3_predictor, image: np.ndarray, text_prompt: str):
#     """Segment image using SAM 3 text prompt"""
#     from sam3.detector import SAM3Detector
    
#     sam3_predictor.set_image(image)
#     detector = SAM3Detector(sam3_predictor.model)
    
#     masks, scores, _ = detector.predict_text(
#         image=image,
#         text_prompt=text_prompt,
#     )
    
#     # Combine all instances
#     combined_mask = np.zeros_like(masks[0], dtype=bool)
#     for mask in masks:
#         combined_mask = combined_mask | mask
    
#     return combined_mask

def segment_with_sam3(
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    sam3_device: str,
    image: np.ndarray,
    text_prompt: str,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """
    Segment image using SAM3 with a text prompt (Transformers version).

    Args:
        sam3_model: Loaded Sam3Model.
        sam3_processor: Loaded Sam3Processor.
        sam3_device: "cuda" or "cpu".
        image: HxWx3 numpy array (RGB).
        text_prompt: e.g. "bicycle", "table", etc.
        score_threshold: confidence threshold for instance selection.
        mask_threshold: binarization threshold for masks.

    Returns:
        combined_mask: boolean mask [H, W] where True = segmented object.
    """
    # Ensure PIL image for the processor
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Build inputs (text-only prompts, like in the model card)
    inputs = sam3_processor(
        images=pil_image,
        text=text_prompt,
        return_tensors="pt",
    ).to(sam3_device)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    # Post-process to instance segmentation masks at original resolution
    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]          # Tensor [num_instances, H, W] (binary)
    if masks.shape[0] == 0:
        # No objects found -> empty mask
        h, w = pil_image.size[1], pil_image.size[0]
        return np.zeros((h, w), dtype=bool)

    # Union of all instance masks
    combined = masks.any(dim=0).cpu().numpy()   # [H, W] bool
    return combined



def process_piebench_image(
    image_id: str,
    annotation: Dict,
    data_dir: str,
    output_dir: str,
    sam3_model,
    sam3_processor,
    sam3_device: str,
    inpaint_pipeline,
    use_sam3: bool = True,
    use_gt_mask: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
):
    """
    Process a single PIE-Bench image
    
    Args:
        image_id: Image ID (e.g., "000000000000")
        annotation: Annotation dict for this image
        data_dir: PIE-Bench data directory
        output_dir: Output directory
        sam3_predictor: SAM 3 predictor
        inpaint_pipeline: SD inpainting pipeline
        use_sam3: If True, use SAM 3 segmentation (default and recommended)
        use_gt_mask: If True, use ground truth mask from PIE-Bench (baseline only).
                     Note: GT masks are primarily for evaluation, not for editing.
        num_inference_steps: SD inference steps
        guidance_scale: CFG scale
        seed: Random seed
    
    Returns:
        Result image
    """
    # Load image
    image_path = os.path.join(data_dir, "annotation_images", annotation["image_path"])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Get mask
    if use_gt_mask:
        # Use ground truth mask from PIE-Bench (baseline comparison only)
        # Note: GT masks are primarily intended for evaluation, not for editing
        mask = decode_rle_mask(annotation["mask"])
    else:
        # Use SAM 3 segmentation (default and recommended workflow)
        if use_sam3:
            # Extract object name from blended_word
            object_name = extract_edit_object(
                annotation["editing_instruction"],
                annotation["blended_word"]
            )
            
            # Segment with SAM 3
            mask = segment_with_sam3(
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
                sam3_device=sam3_device,
                image=image_np,
                text_prompt=object_name,
            )
        else:
            raise ValueError("Must use either SAM 3 or ground truth mask")
    
    # Convert mask to PIL
    mask_pil = Image.fromarray((mask.astype(np.uint8) * 255))
    
    # Get editing prompt (use editing_instruction)
    editing_prompt = annotation["editing_instruction"]
    
    # Set seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device=inpaint_pipeline.device).manual_seed(seed)
    
    # Inpaint
    edited_image = inpaint_pipeline(
        prompt=editing_prompt,
        image=image,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # ---- Build 4-panel composite to match InstructPix2Pix ----
    # 1) text panel: "edit prompt: <instruction>"
    text_panel = txt_draw(f"edit prompt: {editing_prompt}")  # PIL.Image

    # Convert everything to numpy arrays
    text_np = np.array(text_panel)                 # [H, W, 3]
    orig_np = np.array(image)                     # [H, W, 3]
    edited_np = np.array(edited_image)            # [H, W, 3]

    # Sanity: PIE-Bench images are 512x512, SD keeps size, txt_draw usually same height.
    # Make a black spacer with same shape as text panel
    black_np = np.zeros_like(text_np)

    # Concatenate: [text | original | black | edited]
    composite_np = np.concatenate(
        (text_np, orig_np, black_np, edited_np),
        axis=1
    )
    composite_img = Image.fromarray(composite_np.astype(np.uint8))

    return composite_img, mask



def main():
    parser = argparse.ArgumentParser(description='Process PIE-Bench with SAM 3 + SD Inpainting')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='PIE-Bench data directory (contains annotation_images/ and mapping_file.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    # Model arguments
    parser.add_argument('--sam3_checkpoint', type=str, default='models/sam3.pt',
                        help='Path to SAM 3 checkpoint')
    parser.add_argument('--inpaint_model', type=str, default='runwayml/stable-diffusion-inpainting',
                        help='HuggingFace model ID for inpainting')
    
    # Method arguments
    parser.add_argument('--use_sam3', action='store_true', default=True,
                        help='Use SAM 3 for segmentation (default: True, recommended)')
    parser.add_argument('--use_gt_mask', action='store_true',
                        help='Use ground truth mask from PIE-Bench for editing (baseline comparison only). Note: GT masks are primarily for evaluation, not editing.')
    
    # Generation arguments
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # Filter arguments
    parser.add_argument('--edit_category_list', type=int, nargs='+', default=None,
                        help='List of editing categories to process (0-9). If not specified, process all.')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save segmentation masks')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
   # Load models
    print("Loading models...")
    sam3_model, sam3_processor, sam3_device = load_sam3_model(
        model_id="facebook/sam3",   # or make this a CLI arg if you want
        device=args.device,
    )
    inpaint_pipeline = load_inpainting_model(args.inpaint_model, args.device)
    print("Models loaded!\n")

    
    # Load PIE-Bench annotations
    mapping_file = os.path.join(args.data_dir, "mapping_file.json")
    print(f"Loading annotations from {mapping_file}...")
    annotations = load_piebench_annotations(mapping_file)
    print(f"Loaded {len(annotations)} annotations\n")
    
    # Filter by editing category if specified
    if args.edit_category_list is not None:
        category_ids = set(str(c) for c in args.edit_category_list)
        filtered_annotations = {
            k: v for k, v in annotations.items()
            if v["editing_type_id"] in category_ids
        }
        print(f"Filtering to categories {args.edit_category_list}")
        print(f"Filtered to {len(filtered_annotations)} images\n")
        annotations = filtered_annotations
    
    # Limit number of images if specified
    if args.max_images is not None:
        image_ids = list(annotations.keys())[:args.max_images]
        annotations = {k: annotations[k] for k in image_ids}
        print(f"Processing first {args.max_images} images\n")
    
    # Process each image
    results_metadata = {}
    total_time = 0
    
    for i, (image_id, annotation) in enumerate(tqdm(annotations.items(), desc="Processing images"), 1):
        # Create output path maintaining PIE-Bench structure
        rel_path = annotation["image_path"]
        output_path = os.path.join(args.output_dir, "annotation_images", rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            start_time = time.time()
            
            result, mask = process_piebench_image(
                image_id=image_id,
                annotation=annotation,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
                sam3_device=sam3_device,
                inpaint_pipeline=inpaint_pipeline,
                use_sam3=not args.use_gt_mask,
                use_gt_mask=args.use_gt_mask,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Save result
            result.save(output_path)
            
            # Save mask if requested
            if args.save_masks:
                mask_path = output_path.replace('.jpg', '_mask.png')
                Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
            
            # Store metadata
            results_metadata[image_id] = {
                'image_path': rel_path,
                'editing_instruction': annotation['editing_instruction'],
                'editing_type_id': annotation['editing_type_id'],
                'processing_time': elapsed,
                'output_path': output_path,
            }
            
        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "processing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results_metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {len(results_metadata)}")
    print(f"Total time: {total_time:.2f}s")
    if len(results_metadata) > 0:
        print(f"Average time per image: {total_time/len(results_metadata):.2f}s")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print category breakdown
    category_counts = {}
    for meta in results_metadata.values():
        cat = meta['editing_type_id']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nImages per category:")
    for cat in sorted(category_counts.keys(), key=int):
        print(f"  Category {cat}: {category_counts[cat]} images")


if __name__ == '__main__':
    main()