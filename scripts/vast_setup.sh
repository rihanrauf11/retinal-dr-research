#!/bin/bash

# vast_setup.sh
# Setup and verification script for Vast.ai instances
# Run this script ONCE when you first start a new Vast.ai instance
#
# Usage:
#   bash scripts/vast_setup.sh
#
# This script will:
# 1. Verify GPU is available and optimal (RTX 5090 recommended)
# 2. Check all required volume mounts exist
# 3. Verify RETFound weights are present
# 4. Validate data directory structure
# 5. Run a quick sanity test
# 6. Display system information

set -e  # Exit on any error

echo "=================================="
echo "Vast.ai Environment Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

# 1. Check Python and PyTorch installation
echo "1. Checking Python and PyTorch..."
python_version=$(python --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    print_success "PyTorch version: $pytorch_version"
else
    print_error "PyTorch not installed or not working"
    exit 1
fi

# 2. Check CUDA availability
echo ""
echo "2. Checking CUDA and GPU..."
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>&1)
if [ "$cuda_available" = "True" ]; then
    print_success "CUDA is available"

    # Get GPU information
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>&1)
    print_info "Number of GPUs: $gpu_count"

    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
    print_info "GPU: $gpu_name"

    # Check if it's RTX 5090
    if [[ "$gpu_name" == *"5090"* ]]; then
        print_success "Excellent! You're using RTX 5090 (32GB VRAM)"
        print_info "This config is optimized for RTX 5090 with batch_size=64"
    elif [[ "$gpu_name" == *"4090"* ]]; then
        print_warning "You're using RTX 4090 (24GB VRAM)"
        print_info "Consider reducing batch_size to 48 in config"
    elif [[ "$gpu_name" == *"3090"* ]]; then
        print_warning "You're using RTX 3090 (24GB VRAM)"
        print_info "Consider reducing batch_size to 32 in config"
    else
        print_warning "GPU detected: $gpu_name"
        print_info "This config is optimized for RTX 5090"
        print_info "You may need to adjust batch_size based on your GPU memory"
    fi

    # Display GPU memory
    gpu_memory=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')" 2>&1)
    print_info "GPU Memory: ${gpu_memory}GB"
else
    print_error "CUDA is NOT available. This project requires GPU training."
    exit 1
fi

# 3. Check volume mounts
echo ""
echo "3. Checking volume mounts..."

# Check /data
if [ -d "/data" ]; then
    print_success "/data directory exists"

    # Check if APTOS data exists
    if [ -d "/data/aptos" ]; then
        print_success "  └─ /data/aptos exists"

        # Check for train.csv and images
        if [ -f "/data/aptos/train.csv" ]; then
            print_success "     └─ train.csv found"
            num_train=$(wc -l < /data/aptos/train.csv)
            print_info "        └─ Training samples: $((num_train - 1))"
        else
            print_warning "     └─ train.csv NOT found"
        fi

        if [ -d "/data/aptos/train_images" ]; then
            num_images=$(find /data/aptos/train_images -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
            print_success "     └─ train_images directory found ($num_images images)"
        else
            print_warning "     └─ train_images directory NOT found"
        fi
    else
        print_warning "  └─ /data/aptos NOT found"
        print_info "     You need to upload APTOS dataset to /data/aptos/"
    fi
else
    print_error "/data directory NOT mounted"
    print_info "Mount Vast.ai storage to /data"
    exit 1
fi

# Check /models
echo ""
if [ -d "/models" ]; then
    print_success "/models directory exists"

    # Check for RETFound weights
    if [ -f "/models/RETFound_cfp_weights.pth" ]; then
        file_size=$(du -h /models/RETFound_cfp_weights.pth | cut -f1)
        print_success "  └─ RETFound_cfp_weights.pth found (${file_size})"
    else
        print_error "  └─ RETFound_cfp_weights.pth NOT found"
        print_info "     Download from: https://github.com/rmaphoh/RETFound_MAE"
        print_info "     Upload to: /models/RETFound_cfp_weights.pth"
        exit 1
    fi
else
    print_error "/models directory NOT mounted"
    print_info "Mount Vast.ai storage to /models"
    exit 1
fi

# Check /results
echo ""
if [ -d "/results" ]; then
    print_success "/results directory exists"

    # Create subdirectories if they don't exist
    mkdir -p /results/retfound_lora/checkpoints
    mkdir -p /results/retfound_lora/logs
    mkdir -p /results/retfound_lora/tensorboard
    print_success "  └─ Created output directories"
else
    print_error "/results directory NOT mounted"
    print_info "Mount Vast.ai storage to /results"
    exit 1
fi

# 4. Verify key dependencies
echo ""
echo "4. Checking Python dependencies..."
dependencies=("torch" "torchvision" "timm" "transformers" "peft" "albumentations" "pandas" "numpy")

all_installed=true
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)" 2>/dev/null || echo "unknown")
        print_success "$dep ($version)"
    else
        print_error "$dep NOT installed"
        all_installed=false
    fi
done

if [ "$all_installed" = false ]; then
    print_error "Some dependencies are missing"
    exit 1
fi

# 5. Run quick sanity test
echo ""
echo "5. Running quick sanity test..."

# Test: Load RETFound model
print_info "Testing RETFound model loading..."
python -c "
import torch
import sys
sys.path.append('/workspace/dr-classification')

try:
    from scripts.retfound_lora import RETFoundLoRA

    model = RETFoundLoRA(
        checkpoint_path='/models/RETFound_cfp_weights.pth',
        num_classes=5,
        lora_r=16,
        lora_alpha=64,
        head_dropout=0.3
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).cuda()
    model = model.cuda()
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5), f'Expected shape (2, 5), got {output.shape}'
    print('SUCCESS: RETFound model loaded and tested successfully')
except Exception as e:
    print(f'ERROR: {str(e)}')
    sys.exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    print_success "Sanity test passed!"
else
    print_error "Sanity test failed"
    exit 1
fi

# 6. Display system information
echo ""
echo "6. System Information:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Working Directory: $(pwd)"
echo "GPU: $gpu_name"
echo "GPU Memory: ${gpu_memory}GB"
echo "CUDA Version: $(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' || echo 'N/A')"
echo "PyTorch: $pytorch_version"
echo "Python: $python_version"
echo ""
echo "Volume Mounts:"
echo "  /data    → $(du -sh /data 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  /models  → $(du -sh /models 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  /results → $(du -sh /results 2>/dev/null | cut -f1 || echo 'N/A')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 7. Display next steps
echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Start training:"
echo "   bash scripts/vast_train.sh"
echo ""
echo "2. Or run training manually:"
echo "   python3 scripts/train_retfound_lora.py \\"
echo "     --checkpoint_path /models/RETFound_cfp_weights.pth \\"
echo "     --config configs/vastai_retfound_lora.yaml"
echo ""
echo "3. Monitor training:"
echo "   tail -f /results/retfound_lora/logs/training.log"
echo ""
echo "4. Optional - Use screen/tmux for persistent sessions:"
echo "   screen -S training"
echo "   bash scripts/vast_train.sh"
echo "   # Press Ctrl+A then D to detach"
echo "   # Reconnect: screen -r training"
echo ""
