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

### Use ground truth masks (no SAM 3)

```bash
# Use PIE-Bench's provided masks instead of SAM 3
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
    --use_sam3 \                           # Use SAM 3 (default)
    --use_gt_mask \                        # Use PIE-Bench GT masks instead
    
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
   - If `--use_gt_mask`: Use PIE-Bench's ground truth mask
   - Else: Use SAM 3 to segment object (extracted from `blended_word`)
3. **Inpaint** using `editing_instruction` as prompt
4. **Save** result maintaining directory structure

---

## SAM 3 vs Ground Truth Masks

### SAM 3 Mode (default)
```bash
python process_piebench.py --data_dir ./data --output_dir ./output
```

- **Pros:** Tests your SAM 3 segmentation quality
- **Cons:** Segmentation may differ from ground truth

### Ground Truth Mask Mode
```bash
python process_piebench.py --data_dir ./data --output_dir ./output --use_gt_mask
```

- **Pros:** Uses exact masks from PIE-Bench (fair comparison with other methods)
- **Cons:** Not testing SAM 3 segmentation

**Recommendation:** Run both modes to compare!

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

### Use GT masks (baseline)
```bash
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

Metrics include:
- Structure distance
- CLIP similarity
- PSNR/LPIPS/SSIM (unedited parts)
- And more...

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
- Compare `--use_sam3` vs `--use_gt_mask`

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

**To compare fairly:**
Use `--use_gt_mask` to use same masks as baseline methods.

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