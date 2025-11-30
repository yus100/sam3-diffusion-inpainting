#!/bin/bash
# Setup script for SAM 3 + SD Inpainting Pipeline

echo "Setting up SAM 3 + Stable Diffusion Inpainting Pipeline..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install SAM 3 from source
#echo "Installing SAM 3..."
#pip install git+https://github.com/facebookresearch/sam3.git

# Download SAM 3 model
# echo "Downloading SAM 3 model checkpoint..."
# echo "You need to:"
# echo "1. Request access at https://huggingface.co/facebook/sam3"
# echo "2. Download sam3.pt to ./models/"
# echo ""
# echo "Or use HuggingFace CLI:"
# echo "  huggingface-cli login"
# echo "  huggingface-cli download facebook/sam3 sam3.pt --local-dir ./models/"

# echo ""
# echo "Setup complete!"
# echo "Edit sam3_inpaint_pipeline.py to point to your SAM 3 checkpoint."