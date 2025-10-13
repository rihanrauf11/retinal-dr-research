#!/bin/bash

# vast_train.sh
# Training orchestration script for Vast.ai
# Handles training launch, error recovery, and checkpoint management
#
# Usage:
#   bash scripts/vast_train.sh [OPTIONS]
#
# Options:
#   --resume <path>    Resume from checkpoint
#   --wandb            Enable Weights & Biases logging
#   --test-mode        Quick test run (5 epochs, small batch)
#
# Examples:
#   # Standard training
#   bash scripts/vast_train.sh
#
#   # Training with W&B
#   bash scripts/vast_train.sh --wandb
#
#   # Resume from checkpoint
#   bash scripts/vast_train.sh --resume /results/retfound_lora/checkpoints/checkpoint_epoch_10.pth
#
#   # Quick test
#   bash scripts/vast_train.sh --test-mode

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}ℹ${NC} $1"
}

# Parse command line arguments
RESUME_PATH=""
USE_WANDB=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  RETFound + LoRA Training on Vast.ai"
echo "=========================================="
echo ""

# Set working directory
cd /workspace/dr-classification

# Check if setup has been run
if [ ! -f "/models/RETFound_cfp_weights.pth" ]; then
    print_error "RETFound weights not found!"
    print_info "Run: bash scripts/vast_setup.sh first"
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "$RESUME_PATH" ]; then
    print_info "Mode: Resume training"
    print_info "Checkpoint: $RESUME_PATH"
else
    print_info "Mode: Fresh training"
fi

if [ "$USE_WANDB" = true ]; then
    print_info "W&B Logging: Enabled"
    # Check if WANDB_API_KEY is set
    if [ -z "$WANDB_API_KEY" ]; then
        print_warning "WANDB_API_KEY not set"
        print_info "Set it in .env or: export WANDB_API_KEY=your_key"
    fi
else
    print_info "W&B Logging: Disabled"
fi

if [ "$TEST_MODE" = true ]; then
    print_warning "TEST MODE: Running quick test (5 epochs)"
fi

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
print_info "GPU: $GPU_NAME"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create log directory
LOG_DIR="/results/retfound_lora/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# Prepare training command
TRAIN_CMD="python3 scripts/train_retfound_lora.py"
TRAIN_CMD="$TRAIN_CMD --checkpoint_path /models/RETFound_cfp_weights.pth"

# Use appropriate config based on test mode
if [ "$TEST_MODE" = true ]; then
    # Create temporary test config
    TEST_CONFIG="/tmp/test_config.yaml"
    cp configs/vastai_retfound_lora.yaml "$TEST_CONFIG"
    # Modify for quick test (would need yq or python to edit YAML properly)
    TRAIN_CMD="$TRAIN_CMD --config $TEST_CONFIG --epochs 5 --batch_size 16"
    print_info "Using test configuration (5 epochs, batch_size=16)"
else
    TRAIN_CMD="$TRAIN_CMD --config configs/vastai_retfound_lora.yaml"
fi

# Add resume path if specified
if [ -n "$RESUME_PATH" ]; then
    if [ ! -f "$RESUME_PATH" ]; then
        print_error "Checkpoint not found: $RESUME_PATH"
        exit 1
    fi
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
fi

# Add W&B flags if enabled
if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb"
    TRAIN_CMD="$TRAIN_CMD --wandb-project dr-retfound-vastai"
    TRAIN_CMD="$TRAIN_CMD --wandb-run-name retfound_lora_${TIMESTAMP}"
    TRAIN_CMD="$TRAIN_CMD --wandb-tags vastai rtx5090 lora"
fi

# Display final command
echo "Training command:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$TRAIN_CMD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_info "Log file: $LOG_FILE"
echo ""

# Function to handle interrupts
cleanup() {
    echo ""
    print_warning "Training interrupted!"
    print_info "Checkpoint saved to: /results/retfound_lora/checkpoints/checkpoint_interrupted.pth"
    print_info "Resume with: bash scripts/vast_train.sh --resume /results/retfound_lora/checkpoints/checkpoint_interrupted.pth"
    exit 130
}

# Trap Ctrl+C and other signals
trap cleanup INT TERM

# Display training start message
echo "=========================================="
print_success "Starting training..."
echo "=========================================="
echo ""
print_info "Monitor progress:"
print_info "  tail -f $LOG_FILE"
print_info ""
print_info "Check GPU usage:"
print_info "  watch -n 1 nvidia-smi"
print_info ""
print_info "To detach from session:"
print_info "  Press Ctrl+A then D (if using screen)"
print_info "  Press Ctrl+B then D (if using tmux)"
echo ""
echo "Training output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run training with output to both console and log file
$TRAIN_CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_STATUS=${PIPESTATUS[0]}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    print_success "Training completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  Checkpoints: /results/retfound_lora/checkpoints/"
    echo "  Logs:        /results/retfound_lora/logs/"
    echo "  TensorBoard: /results/retfound_lora/tensorboard/"
    echo ""

    # Display best checkpoint
    BEST_CHECKPOINT="/results/retfound_lora/checkpoints/best_model.pth"
    if [ -f "$BEST_CHECKPOINT" ]; then
        print_success "Best model: $BEST_CHECKPOINT"
        # Extract best accuracy if available in log
        BEST_ACC=$(grep -i "best.*accuracy\|saving.*best" "$LOG_FILE" | tail -1 || echo "")
        if [ -n "$BEST_ACC" ]; then
            print_info "  $BEST_ACC"
        fi
    fi

    echo ""
    echo "Next steps:"
    echo "1. Evaluate on test set:"
    echo "   python3 scripts/evaluate_cross_dataset.py \\"
    echo "     --checkpoint /results/retfound_lora/checkpoints/best_model.pth \\"
    echo "     --model_type lora \\"
    echo "     --datasets APTOS:/data/aptos/test.csv:/data/aptos/test_images"
    echo ""
    echo "2. Download results from Vast.ai:"
    echo "   rsync -avz <vast_instance>:/results/ ./local_results/"
    echo ""
    echo "3. View TensorBoard:"
    echo "   tensorboard --logdir /results/retfound_lora/tensorboard"
    echo ""

    if [ "$USE_WANDB" = true ]; then
        print_info "View results on W&B dashboard"
    fi

else
    echo ""
    print_error "Training failed with exit code: $EXIT_STATUS"
    print_info "Check log file: $LOG_FILE"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Reduce batch_size in config"
    echo "  - Data not found: Check volume mounts"
    echo "  - CUDA error: Check GPU availability"
    echo ""
    exit $EXIT_STATUS
fi
