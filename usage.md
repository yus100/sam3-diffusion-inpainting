# PIE-Bench Processing Guide

Process PIE-Bench dataset using SAM 3 + Stable Diffusion Inpainting.

## Quick Start

```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output
```

---

## PIE-Bench Structure

```
data/
├── annotation_images/
│   ├── 0_random_140/          # 140 random edits
│   ├── 1_change_object_80/    # 80 object changes
│   │   ├── 1_artificial/
│   │   └── 2_natural/
│   ├── 2_add_object_80/       # 80 object additions
│   ├── 3_delete_object_80/    # 80 object deletions
│   ├── 4_change_attribute_content_40/
│   ├── 5_change_attribute_pose_40/
│   ├── 6_change_attribute_color_40/
│   ├── 7_change_attribute_material_40/
│   ├── 8_change_background_80/
│   └── 9_change_style_80/
└── mapping_file.json          # Annotations
```

**Total: 700 images across 10 editing types**

---

## Annotation Format

Each entry in `mapping_file.json`:

```json
{
  "000000000000": {
    "image_path": "0_random_140/000000000000.jpg",
    "original_prompt": "a slanted mountain bicycle on the road",
    "editing_prompt": "a slanted rusty mountain bicycle on the road",
    "editing_instruction": "Make the frame of the bike rusty",
    "editing_type_id": "0",
    "blended_word": "bicycle bicycle",
    "mask": {...}  // RLE encoded mask
  }
}
```

---

## Usage Options

### Basic: Process all categories

```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output
```

### Process specific categories

```bash
# Process only categories 0, 1, 2
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output \
    --edit_category_list 0 1 2
```

### Use ground truth masks for baseline comparison (optional)

```bash
# Use PIE-Bench's provided masks instead of SAM 3 (for baseline comparison only)
# Note: GT masks are primarily for evaluation, not for editing
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output \
    --use_gt_mask
```

### Test on small subset

```bash
# Process only first 10 images
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output \
    --max_images 10
```

### Save segmentation masks

```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output \
    --save_masks
```

---

## Full Options

```bash
python process_piebench.py \
    --data_dir ./data \                    # PIE-Bench data directory
    --output_dir ./output \                # Output directory
    
    # Method options
    --use_sam3 \                           # Use SAM 3 for segmentation (default)
    --use_gt_mask \                        # Use PIE-Bench GT masks for editing
    
    # Model options
    --sam3_checkpoint models/sam3.pt \     # SAM 3 model path
    --inpaint_model runwayml/stable-diffusion-inpainting \
    
    # Generation options
    --steps 50 \                           # Inference steps
    --guidance_scale 7.5 \                 # CFG scale
    --seed 42 \                            # Random seed
    
    # Filter options
    --edit_category_list 0 1 2 3 \        # Process specific categories
    --max_images 100 \                     # Limit number of images
    
    # Output options
    --save_masks \                         # Save segmentation masks
    --device cuda                          # Device (cuda/cpu)
```

---

## Editing Categories

| ID | Category | Count | Description |
|----|----------|-------|-------------|
| 0 | Random | 140 | Random prompts |
| 1 | Change object | 80 | Change object to another |
| 2 | Add object | 80 | Add new object |
| 3 | Delete object | 80 | Remove object |
| 4 | Change content | 40 | Change expression/content |
| 5 | Change pose | 40 | Change pose |
| 6 | Change color | 40 | Change color |
| 7 | Change material | 40 | Change material |
| 8 | Change background | 80 | Change background |
| 9 | Change style | 80 | Change artistic style |

---

## Output Structure

The script maintains PIE-Bench's folder structure:

```
output/
├── annotation_images/
│   ├── 0_random_140/
│   │   ├── 000000000000.jpg      # Edited image
│   │   ├── 000000000000_mask.png # Mask (if --save_masks)
│   │   └── ...
│   ├── 1_change_object_80/
│   └── ...
└── processing_metadata.json      # Processing statistics
```

---

## How It Works

For each image:

1. **Load annotation** from `mapping_file.json`
2. **Get mask:**
   - **Default (recommended):** Use SAM 3 to segment object (extracted from `blended_word`)
   - **Optional (baseline only):** Use `--use_gt_mask` to use PIE-Bench's ground truth mask for comparison
3. **Inpaint** using `editing_instruction` as prompt
4. **Save** result maintaining directory structure

**Note:** Ground truth masks in PIE-Bench are primarily used for **evaluation** (to measure how well unedited regions are preserved), not for the editing process itself. The default workflow uses SAM 3 for segmentation.

---

## SAM 3 vs Ground Truth Masks

### SAM 3 Mode (default and recommended)
```bash
python process_piebench.py --data_dir ./data --output_dir ./output
```

- **Purpose:** Uses SAM 3 for text-based segmentation during editing
- **This is the intended workflow** - SAM 3 segments objects based on text prompts

### Ground Truth Mask Mode (baseline comparison only)
```bash
python process_piebench.py --data_dir ./data --output_dir ./output --use_gt_mask
```

- **Purpose:** Uses PIE-Bench's ground truth masks for editing (baseline comparison)
- **Note:** GT masks are primarily intended for **evaluation**, not for editing
- **Use case:** Only if you want to compare results using GT masks vs SAM 3 masks

**Important:** Ground truth masks in PIE-Bench are used by the evaluation script to compute metrics (e.g., `psnr_unedit_part`, `lpips_unedit_part`) that measure how well unedited regions are preserved. They are not meant to be used during the editing process itself.

---

## Example Commands

### Quick test (10 images, fast)
```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output_test \
    --max_images 10 \
    --steps 20
```

### Process category 1 (change object)
```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output_cat1 \
    --edit_category_list 1
```

### High quality, reproducible
```bash
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output_hq \
    --steps 100 \
    --seed 42 \
    --save_masks
```

### Use GT masks for baseline comparison (optional)
```bash
# Note: GT masks are for evaluation, not editing. This is only for comparison.
python process_piebench.py \
    --data_dir ./data \
    --output_dir ./output_gt \
    --use_gt_mask
```

---

## Evaluation

After processing, you can evaluate using PIE-Bench's evaluation script:

```bash
# From PIE-Bench repo
python evaluation/evaluate.py \
    --metrics "structure_distance" "clip_similarity_target_image" \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --result_path results.csv
```

**Note:** The evaluation script uses **ground truth masks** from PIE-Bench annotations to compute metrics. These masks define which regions were edited and which should remain unchanged, allowing metrics like:
- Structure distance
- CLIP similarity
- PSNR/LPIPS/SSIM on unedited parts (using GT masks to identify unedited regions)
- PSNR/LPIPS/SSIM on edited parts (using GT masks to identify edited regions)
- And more...

The GT masks are essential for evaluation because they tell the metrics calculator which parts of the image should be compared.

---

## Output Metadata

`processing_metadata.json` contains:

```json
{
  "000000000000": {
    "image_path": "0_random_140/000000000000.jpg",
    "editing_instruction": "Make the frame of the bike rusty",
    "editing_type_id": "0",
    "processing_time": 12.3,
    "output_path": "output/annotation_images/0_random_140/000000000000.jpg"
  }
}
```

---

## Tips

**Speed:**
- Use `--steps 20` for faster processing
- Use `--max_images 10` to test first
- Process categories separately for parallel processing

**Quality:**
- Use `--steps 100` for best results
- Use `--seed 42` for reproducibility
- Use SAM 3 for segmentation (default) - GT masks are for evaluation only

**Memory:**
- Process in batches using `--edit_category_list`
- Use smaller categories (4-7 have only 40 images each)

---

## Download PIE-Bench

Get PIE-Bench from: https://forms.gle/hVMkTABb4uvZVjme9

Or see: https://github.com/cure-lab/PnPInversion

---

## Comparison with Original PIE-Bench Methods

Original PIE-Bench uses:
- DDIM inversion + Prompt-to-Prompt
- Null-text inversion
- Other diffusion-based methods

**This script uses:**
- SAM 3 text segmentation (no inversion needed!)
- Direct inpainting with editing instruction
- Much simpler pipeline

**Important distinction:**
- **Editing process:** Uses SAM 3 to segment objects (default) or optionally GT masks for baseline comparison
- **Evaluation process:** Uses GT masks to compute metrics (this is what GT masks are designed for)

---

## That's It!

Process all 700 PIE-Bench images:

```bash
python process_piebench.py --data_dir ./data --output_dir ./output
```

Test on 10 images first:

```bash
python process_piebench.py --data_dir ./data --output_dir ./test --max_images 10 --steps 20
```