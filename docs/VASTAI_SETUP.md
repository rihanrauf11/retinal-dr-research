# Vast.ai Deployment Guide for Diabetic Retinopathy Classification

Complete guide for deploying and training RETFound + LoRA models on Vast.ai cloud GPU instances, optimized for RTX 5090.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Cost Estimates](#cost-estimates)
4. [Step 1: Build Docker Image](#step-1-build-docker-image)
5. [Step 2: Prepare Data & Models](#step-2-prepare-data--models)
6. [Step 3: Create Vast.ai Account](#step-3-create-vastai-account)
7. [Step 4: Launch Instance](#step-4-launch-instance)
8. [Step 5: Setup Environment](#step-5-setup-environment)
9. [Step 6: Start Training](#step-6-start-training)
10. [Step 7: Monitor Training](#step-7-monitor-training)
11. [Step 8: Retrieve Results](#step-8-retrieve-results)
12. [Troubleshooting](#troubleshooting)
13. [Cost Optimization Tips](#cost-optimization-tips)
14. [Advanced Usage](#advanced-usage)

---

## Overview

### What is Vast.ai?

Vast.ai is a peer-to-peer GPU rental marketplace that offers:
- **Affordable GPU access**: 50-80% cheaper than AWS/GCP
- **High-end GPUs**: RTX 4090, RTX 5090, A100, H100
- **Flexible billing**: Pay per second
- **Docker-based deployment**: Easy environment setup

### Why Use Vast.ai for This Project?

- **RTX 5090 Optimized**: Our config is tuned for RTX 5090 (32GB VRAM)
- **Fast Training**: ~3-3.5 hours for 20 epochs (vs 5+ hours on RTX 3090)
- **Cost-Effective**: $0.50-1.00/hour for RTX 5090
- **No Local GPU Required**: Train from any machine

### Expected Performance

| GPU | Training Time | Cost/Run | Batch Size | VRAM Usage |
|-----|---------------|----------|------------|------------|
| **RTX 5090** | **3-3.5 hours** | **$1.50-3.50** | **64** | **~12GB** |
| RTX 4090 | 4-5 hours | $2.00-4.00 | 48 | ~14GB |
| RTX 3090 | 5-6 hours | $1.50-3.00 | 32 | ~12GB |
| A100 (40GB) | 4-5 hours | $5.00-8.00 | 64 | ~14GB |

---

## Prerequisites

### Local Machine Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Hub Account**: Free account ([Sign up](https://hub.docker.com/))
- **Disk Space**: 10GB for Docker image
- **Internet**: Stable connection for uploading data

### Required Files

- **RETFound Weights**: `RETFound_cfp_weights.pth` (~1.5GB)
  - Download from: https://github.com/rmaphoh/RETFound_MAE
- **APTOS Dataset**: Train/test images and CSV files (~3.5GB)
  - Download via: `python scripts/prepare_data.py --aptos-only`

### Accounts Needed

1. **Vast.ai Account**: [Sign up](https://vast.ai/)
2. **Docker Hub Account**: [Sign up](https://hub.docker.com/)
3. **(Optional) Weights & Biases**: For experiment tracking

---

## Cost Estimates

### Training Costs (20 epochs, APTOS dataset)

| GPU | Hourly Rate | Training Time | Total Cost |
|-----|-------------|---------------|------------|
| RTX 5090 | $0.50-1.00/hr | 3.5 hours | **$1.75-3.50** |
| RTX 4090 | $0.40-0.80/hr | 5 hours | $2.00-4.00 |
| RTX 3090 | $0.30-0.60/hr | 6 hours | $1.80-3.60 |
| A100 40GB | $1.00-1.50/hr | 4.5 hours | $4.50-6.75 |

### Storage Costs

- **Data Storage**: $0.10-0.20/GB/month
- **APTOS Dataset**: ~3.5GB = $0.35-0.70/month
- **RETFound Weights**: ~1.5GB = $0.15-0.30/month
- **Results/Checkpoints**: ~2GB = $0.20-0.40/month

### Total Cost Estimate

**Single Training Run on RTX 5090**: $1.75-3.50
**Monthly Storage (if kept)**: $0.70-1.40

**💡 Tip**: Delete data after experiments to save on storage costs.

---

## Step 1: Build Docker Image

### 1.1 Clone Repository

```bash
git clone https://github.com/your-username/retinal-dr-research.git
cd retinal-dr-research
```

### 1.2 Build Docker Image

**Option A: Using Makefile (Recommended)**

```bash
make docker-build DOCKER_USERNAME=your-dockerhub-username
```

**Option B: Using build script**

```bash
bash scripts/build_and_push.sh --username your-dockerhub-username
```

**Option C: Manual Docker command**

```bash
docker build -t your-dockerhub-username/dr-retfound-lora:latest .
```

### 1.3 Test Image Locally (Optional)

```bash
# Test without GPU
docker run --rm your-dockerhub-username/dr-retfound-lora:latest \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test with GPU (requires nvidia-docker)
docker run --gpus all --rm your-dockerhub-username/dr-retfound-lora:latest \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.4 Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push image
make docker-push DOCKER_USERNAME=your-dockerhub-username

# Or manually
docker push your-dockerhub-username/dr-retfound-lora:latest
```

**Expected Output:**
```
The push refers to repository [docker.io/your-username/dr-retfound-lora]
latest: digest: sha256:abc123... size: 4567
```

---

## Step 2: Prepare Data & Models

### 2.1 Download RETFound Weights

1. Visit: https://github.com/rmaphoh/RETFound_MAE
2. Download `RETFound_cfp_weights.pth` (~1.5GB)
3. Save locally (you'll upload to Vast.ai later)

### 2.2 Download APTOS Dataset

```bash
# Setup Kaggle API
pip install kaggle

# Place kaggle.json in ~/.kaggle/
# Get from: https://www.kaggle.com/settings -> API -> Create New Token

# Download APTOS
python scripts/prepare_data.py --aptos-only
```

### 2.3 Verify Data Structure

```bash
data/aptos/
├── train_images/          # 3,662 images
├── test_images/           # 1,928 images
├── train.csv              # Training labels
└── test.csv               # Test labels
```

---

## Step 3: Create Vast.ai Account

### 3.1 Sign Up

1. Go to: https://vast.ai/
2. Click "Sign Up"
3. Verify email

### 3.2 Add Payment Method

1. Go to: https://vast.ai/console/billing/
2. Add credit card or crypto
3. Add initial credits ($10-20 recommended)

### 3.3 Generate SSH Key (if you don't have one)

```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Display public key
cat ~/.ssh/id_rsa.pub
```

### 3.4 Add SSH Key to Vast.ai

1. Go to: https://vast.ai/console/account/
2. Click "SSH Keys"
3. Click "Add SSH Key"
4. Paste your public key (`~/.ssh/id_rsa.pub`)
5. Save

---

## Step 4: Launch Instance

### 4.1 Search for GPU

1. Go to: https://vast.ai/console/create/
2. Click "GPU Rentals"
3. **Filters**:
   - **GPU**: RTX 5090 (or RTX 4090/3090)
   - **VRAM**: ≥24GB (32GB for RTX 5090)
   - **Disk Space**: ≥50GB
   - **Upload Speed**: ≥100 Mbps (for faster data upload)
   - **Reliability Score**: ≥95%

### 4.2 Select Instance

**Look for:**
- ✅ Green "AVAILABLE" status
- ✅ High reliability score (>95%)
- ✅ Good upload/download speed
- ✅ Reasonable price ($0.50-1.00/hr for RTX 5090)

**Sort by**: Price/Performance or Reliability

### 4.3 Configure Instance

1. **Docker Image**: `your-dockerhub-username/dr-retfound-lora:latest`
2. **Disk Space**: 50GB minimum
3. **On-Start Script** (optional):
   ```bash
   # Auto-run setup on start
   cd /workspace/dr-classification
   bash scripts/vast_setup.sh
   ```

### 4.4 Launch

1. Click "Rent"
2. Confirm rental
3. Wait for instance to start (1-2 minutes)

**Instance will be in "loading" state initially, then "running"**

---

## Step 5: Setup Environment

### 5.1 Connect to Instance

```bash
# Get SSH command from Vast.ai console
# Example:
ssh -p 12345 root@123.456.789.10

# Or use Vast.ai CLI
./vast ssh <instance_id>
```

### 5.2 Upload Data to Instance

**Option A: SCP (Secure Copy)**

```bash
# Upload APTOS dataset
scp -P 12345 -r data/aptos root@123.456.789.10:/data/

# Upload RETFound weights
scp -P 12345 models/RETFound_cfp_weights.pth root@123.456.789.10:/models/
```

**Option B: rsync (Faster, resumable)**

```bash
# Upload APTOS dataset
rsync -avz -e "ssh -p 12345" data/aptos/ root@123.456.789.10:/data/aptos/

# Upload RETFound weights
rsync -avz -e "ssh -p 12345" models/RETFound_cfp_weights.pth root@123.456.789.10:/models/
```

**Option C: Direct download on instance**

```bash
# SSH into instance
ssh -p 12345 root@123.456.789.10

# Download APTOS
cd /data
python /workspace/dr-classification/scripts/prepare_data.py --aptos-only

# Download RETFound (if publicly available)
cd /models
wget -O RETFound_cfp_weights.pth <URL_IF_PUBLIC>
```

### 5.3 Run Setup Script

```bash
# Inside instance
cd /workspace/dr-classification
bash scripts/vast_setup.sh
```

**Expected Output:**
```
==================================
Vast.ai Environment Setup
==================================

1. Checking Python and PyTorch...
✓ Python version: 3.10.x
✓ PyTorch version: 2.1.0+cu121

2. Checking CUDA and GPU...
✓ CUDA is available
ℹ Number of GPUs: 1
ℹ GPU: NVIDIA GeForce RTX 5090
✓ Excellent! You're using RTX 5090 (32GB VRAM)
ℹ GPU Memory: 32.0GB

3. Checking volume mounts...
✓ /data directory exists
✓   └─ /data/aptos exists
✓      └─ train.csv found
✓      └─ train_images directory found (3662 images)

✓ /models directory exists
✓   └─ RETFound_cfp_weights.pth found (1.4G)

✓ /results directory exists
✓   └─ Created output directories

4. Checking Python dependencies...
✓ torch (2.1.0)
✓ timm (0.9.12)
✓ transformers (4.35.0)
✓ peft (0.6.0)
...

5. Running quick sanity test...
ℹ Testing RETFound model loading...
✓ Sanity test passed!

==================================
✓ Setup Complete!
==================================
```

---

## Step 6: Start Training

### 6.1 Start Training Session

**Option A: Interactive (see output in real-time)**

```bash
cd /workspace/dr-classification
bash scripts/vast_train.sh
```

**Option B: With W&B logging**

```bash
# Set W&B API key first
export WANDB_API_KEY=your_wandb_api_key

bash scripts/vast_train.sh --wandb
```

**Option C: In background with screen**

```bash
# Start screen session
screen -S training

# Start training
bash scripts/vast_train.sh

# Detach: Press Ctrl+A then D
# Reattach: screen -r training
```

**Option D: In background with tmux**

```bash
# Start tmux session
tmux new -s training

# Start training
bash scripts/vast_train.sh

# Detach: Press Ctrl+B then D
# Reattach: tmux attach -t training
```

### 6.2 Training Output

```
==========================================
  RETFound + LoRA Training on Vast.ai
==========================================

Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ Mode: Fresh training
ℹ W&B Logging: Enabled
ℹ GPU: NVIDIA GeForce RTX 5090
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training command:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python3 scripts/train_retfound_lora.py --checkpoint_path /models/RETFound_cfp_weights.pth --config configs/vastai_retfound_lora.yaml --wandb
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

==========================================
✓ Starting training...
==========================================

Loading RETFound model from /models/RETFound_cfp_weights.pth...
✓ Model loaded successfully

Applying LoRA with r=16, alpha=64...
trainable params: 1,585,413 || all params: 303,804,170 || trainable%: 0.52%

Loading dataset from /data/aptos/train.csv...
✓ Loaded 3,662 images (5 classes)
✓ Train: 2,930 images, Val: 732 images

Starting training for 20 epochs...

Epoch 1/20:
100%|█████████████| 46/46 [02:15<00:00, 2.94s/batch, loss=1.234, acc=0.654]
Validation: 100%|██████| 12/12 [00:18<00:00, 1.53s/batch]
Epoch 1 - Train Loss: 1.234, Train Acc: 65.4% | Val Loss: 1.098, Val Acc: 70.2%
✓ New best model saved! (Val Acc: 70.2%)

Epoch 2/20:
...
```

---

## Step 7: Monitor Training

### 7.1 Real-Time Monitoring

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Expected GPU usage (RTX 5090):**
- **GPU Utilization**: 90-100%
- **Memory Usage**: ~12GB / 32GB
- **Temperature**: 60-75°C
- **Power**: 300-400W

**Monitor training logs:**
```bash
tail -f /results/retfound_lora/logs/training_*.log
```

### 7.2 Weights & Biases Dashboard

If you enabled W&B:

1. Go to: https://wandb.ai/
2. Navigate to your project: `dr-retfound-vastai`
3. View real-time metrics:
   - Training/validation loss
   - Accuracy
   - Learning rate
   - GPU utilization
   - Training time

### 7.3 TensorBoard (Alternative)

```bash
# On Vast.ai instance
tensorboard --logdir /results/retfound_lora/tensorboard --port 6006

# Forward port to local machine
ssh -p 12345 -L 6006:localhost:6006 root@123.456.789.10

# Open in browser
http://localhost:6006
```

### 7.4 Training Progress Checklist

After ~30 minutes, check:
- ✅ Training loss decreasing
- ✅ Validation accuracy increasing
- ✅ GPU utilization >90%
- ✅ No out-of-memory errors
- ✅ Checkpoints being saved to `/results/`

---

## Step 8: Retrieve Results

### 8.1 Training Completion

When training finishes, you'll see:

```
==========================================
✓ Training completed successfully!
==========================================

Results saved to:
  Checkpoints: /results/retfound_lora/checkpoints/
  Logs:        /results/retfound_lora/logs/
  TensorBoard: /results/retfound_lora/tensorboard/

✓ Best model: /results/retfound_lora/checkpoints/best_model.pth
  Epoch 15 - Val Acc: 88.5%
```

### 8.2 Download Results

**Option A: SCP**

```bash
# Download all results
scp -P 12345 -r root@123.456.789.10:/results/retfound_lora ./local_results/

# Download only checkpoints
scp -P 12345 root@123.456.789.10:/results/retfound_lora/checkpoints/best_model.pth ./
```

**Option B: rsync (Recommended)**

```bash
# Download all results
rsync -avz -e "ssh -p 12345" root@123.456.789.10:/results/retfound_lora/ ./local_results/

# Resume interrupted transfer
rsync -avz --partial -e "ssh -p 12345" root@123.456.789.10:/results/ ./local_results/
```

**Option C: Vast.ai File Browser**

1. Go to Vast.ai console
2. Click on your instance
3. Click "Files"
4. Navigate to `/results/retfound_lora/checkpoints/`
5. Download `best_model.pth`

### 8.3 Verify Downloaded Files

```bash
local_results/
├── checkpoints/
│   ├── best_model.pth           # Best model (~1.2GB)
│   ├── checkpoint_epoch_5.pth   # Intermediate checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── training_history.json    # Training metrics
├── logs/
│   └── training_20240101_120000.log
└── tensorboard/
    └── events.out.tfevents.*
```

### 8.4 Stop Instance

**IMPORTANT**: Stop or destroy instance to avoid charges!

```bash
# From Vast.ai console
1. Go to: https://vast.ai/console/instances/
2. Click "Stop" or "Destroy" on your instance

# Or via CLI
./vast stop instance <instance_id>
./vast destroy instance <instance_id>
```

**💰 Cost Tip**: Destroy instance if you're done. Stop if you want to keep storage.

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Reduce batch size in config**:
   ```yaml
   # Edit configs/vastai_retfound_lora.yaml
   training:
     batch_size: 32  # Reduce from 64
   ```

2. **Reduce LoRA rank**:
   ```yaml
   model:
     lora_r: 8  # Reduce from 16
   ```

3. **Enable gradient checkpointing** (if implemented)

4. **Use smaller images** (if implemented)

### Issue: Data Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/data/aptos/train.csv'
```

**Solutions:**

1. Verify data upload:
   ```bash
   ls -lh /data/aptos/
   ```

2. Re-upload data:
   ```bash
   scp -P 12345 -r data/aptos root@IP:/data/
   ```

3. Check volume mounts in Vast.ai instance settings

### Issue: RETFound Weights Not Found

**Error:**
```
FileNotFoundError: /models/RETFound_cfp_weights.pth not found
```

**Solutions:**

1. Verify weights exist:
   ```bash
   ls -lh /models/
   ```

2. Upload weights:
   ```bash
   scp -P 12345 models/RETFound_cfp_weights.pth root@IP:/models/
   ```

3. Check file size (~1.4GB):
   ```bash
   du -h /models/RETFound_cfp_weights.pth
   ```

### Issue: Slow Training

**Symptom**: Training slower than expected

**Solutions:**

1. **Check GPU utilization**:
   ```bash
   nvidia-smi
   # Should be 90-100%
   ```

2. **Increase num_workers**:
   ```yaml
   data:
     num_workers: 8  # Increase if CPU has capacity
   ```

3. **Enable mixed precision** (should already be enabled):
   ```yaml
   training:
     mixed_precision: true
   ```

4. **Check disk I/O** (slow storage can bottleneck):
   ```bash
   iostat -x 1
   ```

### Issue: SSH Connection Lost

**Symptom**: Training interrupted due to SSH disconnection

**Solutions:**

1. **Use screen or tmux** (recommended):
   ```bash
   screen -S training
   bash scripts/vast_train.sh
   # Press Ctrl+A then D to detach
   ```

2. **Enable autosave** (already implemented):
   - Training automatically saves checkpoint on interruption

3. **Resume training**:
   ```bash
   bash scripts/vast_train.sh --resume /results/retfound_lora/checkpoints/checkpoint_interrupted.pth
   ```

### Issue: W&B Not Logging

**Symptom**: No data appearing in W&B dashboard

**Solutions:**

1. **Set W&B API key**:
   ```bash
   export WANDB_API_KEY=your_key
   ```

2. **Check W&B mode**:
   ```bash
   echo $WANDB_MODE
   # Should be "online"
   ```

3. **Test W&B connection**:
   ```bash
   python -c "import wandb; wandb.login()"
   ```

4. **Check firewall** (W&B needs internet access)

---

## Cost Optimization Tips

### 1. Choose the Right GPU

| GPU | Best For | Cost/Hour |
|-----|----------|-----------|
| **RTX 5090** | **Fastest training, best price/performance** | $0.50-1.00 |
| RTX 4090 | Good balance | $0.40-0.80 |
| RTX 3090 | Budget option | $0.30-0.60 |
| A100 | Overkill for this task | $1.00-1.50 |

**Recommendation**: RTX 5090 for best value (fastest + cheapest per run)

### 2. Minimize Storage Costs

```bash
# Delete data after training
rm -rf /data/aptos

# Keep only best checkpoint
rm /results/retfound_lora/checkpoints/checkpoint_epoch_*.pth
# Keep: best_model.pth only
```

### 3. Use Spot Instances

- Enable "Interruptible" instances for 50% discount
- Risk: May be interrupted, but can resume from checkpoint

### 4. Stop Instance Immediately After Training

```bash
# Automate shutdown after training
bash scripts/vast_train.sh && sudo shutdown -h now
```

### 5. Batch Multiple Experiments

If running multiple experiments:

```bash
# Train multiple configs in sequence
bash scripts/vast_train.sh --config config1.yaml
bash scripts/vast_train.sh --config config2.yaml
bash scripts/vast_train.sh --config config3.yaml
```

### 6. Monitor Usage

Check your Vast.ai billing:
1. Go to: https://vast.ai/console/billing/
2. Review hourly costs
3. Set billing alerts

---

## Advanced Usage

### Resume Training from Checkpoint

```bash
bash scripts/vast_train.sh --resume /results/retfound_lora/checkpoints/checkpoint_epoch_10.pth
```

### Run Hyperparameter Search

```bash
python3 scripts/hyperparameter_search.py \
    --checkpoint-path /models/RETFound_cfp_weights.pth \
    --data-csv /data/aptos/train.csv \
    --data-img-dir /data/aptos/train_images \
    --n-trials 20 \
    --wandb
```

### Cross-Dataset Evaluation

```bash
# Upload Messidor dataset
scp -P 12345 -r data/messidor root@IP:/data/

# Evaluate
python3 scripts/evaluate_cross_dataset.py \
    --checkpoint /results/retfound_lora/checkpoints/best_model.pth \
    --model_type lora \
    --datasets \
        APTOS:/data/aptos/test.csv:/data/aptos/test_images \
        Messidor:/data/messidor/test.csv:/data/messidor/images
```

### Run Multiple Experiments in Parallel

If your GPU has headroom (RTX 5090 has 32GB):

```bash
# Terminal 1
python3 scripts/train_retfound_lora.py --config config1.yaml &

# Terminal 2
python3 scripts/train_retfound_lora.py --config config2.yaml &
```

**⚠️ Warning**: Monitor GPU memory usage carefully!

---

## Quick Reference

### Essential Commands

```bash
# Build Docker image
make docker-build DOCKER_USERNAME=your-username

# Push Docker image
make docker-push DOCKER_USERNAME=your-username

# Connect to instance
ssh -p PORT root@IP

# Upload data
rsync -avz -e "ssh -p PORT" data/aptos/ root@IP:/data/aptos/

# Run setup
bash scripts/vast_setup.sh

# Start training
bash scripts/vast_train.sh --wandb

# Monitor GPU
watch -n 1 nvidia-smi

# Download results
rsync -avz -e "ssh -p PORT" root@IP:/results/ ./local_results/

# Stop instance (IMPORTANT!)
# Do this from Vast.ai console
```

### Directory Structure on Vast.ai

```
/workspace/dr-classification/    # Code (from Docker image)
├── scripts/
├── configs/
├── docs/
└── ...

/data/                           # Mounted volume
└── aptos/
    ├── train_images/
    ├── train.csv
    └── test.csv

/models/                         # Mounted volume
└── RETFound_cfp_weights.pth

/results/                        # Mounted volume
└── retfound_lora/
    ├── checkpoints/
    ├── logs/
    └── tensorboard/
```

---

## Support

### Getting Help

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/your-username/retinal-dr-research/issues)
- **Vast.ai Support**: https://vast.ai/faq
- **W&B Documentation**: https://docs.wandb.ai/

### Useful Resources

- **Vast.ai Documentation**: https://vast.ai/docs/
- **Docker Documentation**: https://docs.docker.com/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **RETFound Paper**: https://www.nature.com/articles/s41586-023-06555-x

---

**🎉 Congratulations!** You're now ready to train RETFound + LoRA models on Vast.ai with RTX 5090!

**Expected Results**:
- Training time: 3-3.5 hours
- Cost: $1.75-3.50 per run
- Validation accuracy: ~88-89%
- Best model saved for deployment

---

*Last Updated: 2025-10-13*
*Version: 1.0.0*
