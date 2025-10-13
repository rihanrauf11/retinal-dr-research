# Vast.ai Quick Start Guide

**Ultra-fast setup guide for RTX 5090 training on Vast.ai**

Time to first training run: **~15 minutes**

---

## Prerequisites

- Docker installed locally
- Docker Hub account
- Vast.ai account with credits ($10-20)
- RETFound weights downloaded
- APTOS dataset downloaded

---

## 5-Step Setup

### Step 1: Build & Push Docker Image (5 min)

```bash
# Build
make docker-build DOCKER_USERNAME=your-username

# Push
make docker-push DOCKER_USERNAME=your-username
```

### Step 2: Launch Vast.ai Instance (2 min)

1. Go to: https://vast.ai/console/create/
2. Search for **RTX 5090**
3. Select instance with:
   - ✅ 32GB VRAM
   - ✅ 50GB+ disk
   - ✅ >95% reliability
4. Configure:
   - **Docker Image**: `your-username/dr-retfound-lora:latest`
   - **Disk**: 50GB
5. Click "Rent"

### Step 3: Upload Data (5 min)

```bash
# Get SSH details from Vast.ai console
PORT=12345
IP=123.456.789.10

# Upload data
rsync -avz -e "ssh -p $PORT" data/aptos/ root@$IP:/data/aptos/
rsync -avz -e "ssh -p $PORT" models/RETFound_cfp_weights.pth root@$IP:/models/
```

### Step 4: Setup Environment (1 min)

```bash
# SSH into instance
ssh -p $PORT root@$IP

# Run setup
cd /workspace/dr-classification
bash scripts/vast_setup.sh
```

### Step 5: Start Training (2 min)

```bash
# Option A: Simple training
bash scripts/vast_train.sh

# Option B: With W&B logging (recommended)
export WANDB_API_KEY=your_key
bash scripts/vast_train.sh --wandb

# Option C: In background (recommended)
screen -S training
bash scripts/vast_train.sh --wandb
# Press Ctrl+A then D to detach
```

---

## Training Time & Cost

**RTX 5090 (32GB VRAM)**:
- ⏱️ Training time: **3-3.5 hours** (20 epochs)
- 💰 Cost: **$1.75-3.50** per run
- 📊 Expected accuracy: **~88-89%**

---

## Monitor Training

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f /results/retfound_lora/logs/training_*.log

# W&B dashboard
# Visit: https://wandb.ai/your-username/dr-retfound-vastai
```

---

## Download Results

```bash
# After training completes (3-3.5 hours)
rsync -avz -e "ssh -p $PORT" root@$IP:/results/retfound_lora/ ./results/
```

---

## Stop Instance (IMPORTANT!)

**Don't forget to stop/destroy instance to avoid charges!**

1. Go to: https://vast.ai/console/instances/
2. Click **"Destroy"** on your instance

---

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size in configs/vastai_retfound_lora.yaml
training:
  batch_size: 32  # Down from 64
```

### Data Not Found
```bash
# Check data exists
ls -lh /data/aptos/
ls -lh /models/

# Re-upload if needed
rsync -avz -e "ssh -p $PORT" data/aptos/ root@$IP:/data/aptos/
```

### Training Interrupted
```bash
# Resume from last checkpoint
bash scripts/vast_train.sh --resume /results/retfound_lora/checkpoints/checkpoint_interrupted.pth
```

---

## What You Get

After 3-3.5 hours:

- ✅ Trained RETFound + LoRA model
- ✅ Best checkpoint (~1.2GB)
- ✅ Training history (JSON)
- ✅ TensorBoard logs
- ✅ W&B experiment logs (if enabled)
- ✅ ~88-89% validation accuracy

---

## Next Steps

1. **Download best model**:
   ```bash
   scp -P $PORT root@$IP:/results/retfound_lora/checkpoints/best_model.pth ./
   ```

2. **Evaluate on test set**:
   ```bash
   python3 scripts/evaluate_cross_dataset.py \
       --checkpoint best_model.pth \
       --model_type lora \
       --datasets APTOS:data/aptos/test.csv:data/aptos/test_images
   ```

3. **Deploy model** for inference

---

## Complete Documentation

For detailed instructions, see:
- **[docs/VASTAI_SETUP.md](docs/VASTAI_SETUP.md)** - Complete setup guide
- **[README.md](README.md)** - Project overview
- **[CLAUDE.md](CLAUDE.md)** - Development guide

---

## Quick Commands Reference

```bash
# Build Docker
make docker-build DOCKER_USERNAME=username

# Push Docker
make docker-push DOCKER_USERNAME=username

# Upload data
rsync -avz -e "ssh -p PORT" data/aptos/ root@IP:/data/aptos/

# SSH connect
ssh -p PORT root@IP

# Setup
bash scripts/vast_setup.sh

# Train
bash scripts/vast_train.sh --wandb

# Monitor
watch -n 1 nvidia-smi

# Download results
rsync -avz -e "ssh -p PORT" root@IP:/results/ ./results/
```

---

**Questions?** See [docs/VASTAI_SETUP.md](docs/VASTAI_SETUP.md) for detailed troubleshooting.

**Ready to train!** 🚀
