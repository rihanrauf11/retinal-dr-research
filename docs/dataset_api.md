# Dataset API Documentation

## Overview

The `scripts.dataset` module provides the `RetinalDataset` class, a PyTorch Dataset implementation for loading and processing diabetic retinopathy retinal fundus images. This dataset class handles image loading, label management, and seamless integration with data augmentation pipelines.

**Key Features:**
- Flexible image format support (.png, .jpg, .jpeg)
- Automatic error handling and validation
- Support for both torchvision and albumentations transforms
- Class distribution analysis
- Comprehensive error messages for debugging

**Source:** [scripts/dataset.py](../scripts/dataset.py)

---

## Table of Contents

1. [RetinalDataset Class](#retinaldataset-class)
2. [Class Methods](#class-methods)
   - [\_\_init\_\_](#__init__)
   - [\_\_len\_\_](#__len__)
   - [\_\_getitem\_\_](#__getitem__)
   - [get_class_distribution](#get_class_distribution)
   - [get_sample_info](#get_sample_info)
3. [Usage Examples](#usage-examples)
4. [Integration with DataLoader](#integration-with-dataloader)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## RetinalDataset Class

### Class Signature

```python
class RetinalDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for diabetic retinopathy retinal fundus images."""
```

### Diabetic Retinopathy Severity Levels

The dataset handles 5-class classification:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No Diabetic Retinopathy |
| 1 | Mild NPDR | Mild Non-Proliferative Diabetic Retinopathy |
| 2 | Moderate NPDR | Moderate Non-Proliferative Diabetic Retinopathy |
| 3 | Severe NPDR | Severe Non-Proliferative Diabetic Retinopathy |
| 4 | PDR | Proliferative Diabetic Retinopathy |

### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `csv_file` | `Path` | Path object to the CSV metadata file |
| `img_dir` | `Path` | Path object to the image directory |
| `transform` | `Callable` or `None` | Optional transformation pipeline |
| `data_frame` | `pd.DataFrame` | DataFrame containing image metadata |
| `VALID_EXTENSIONS` | `List[str]` | Supported image extensions: `['.png', '.jpg', '.jpeg']` |
| `MIN_DIAGNOSIS` | `int` | Minimum valid diagnosis value: `0` |
| `MAX_DIAGNOSIS` | `int` | Maximum valid diagnosis value: `4` |

---

## Class Methods

### \_\_init\_\_

Initialize the RetinalDataset with image directory and CSV metadata.

**Signature:**
```python
def __init__(
    self,
    csv_file: str,
    img_dir: str,
    transform: Optional[Callable] = None
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_file` | `str` or `Path` | Required | Path to CSV file with columns `id_code` (image ID) and `diagnosis` (0-4) |
| `img_dir` | `str` or `Path` | Required | Directory containing retinal images named by `id_code` |
| `transform` | `Callable` or `None` | `None` | Optional transform/augmentation pipeline (torchvision or albumentations) |

**CSV File Format:**

The CSV file must contain at minimum:
```csv
id_code,diagnosis
image_001,0
image_002,1
image_003,2
```

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If `csv_file` or `img_dir` does not exist |
| `ValueError` | If required columns (`id_code`, `diagnosis`) are missing from CSV |
| `ValueError` | If diagnosis values are outside the valid range [0, 4] |
| `ValueError` | If CSV file cannot be read (corrupted or invalid format) |

**Example:**

```python
from scripts.dataset import RetinalDataset

# Basic initialization without transforms
dataset = RetinalDataset(
    csv_file='data/aptos/train.csv',
    img_dir='data/aptos/train_images'
)

print(f"Dataset size: {len(dataset)}")
# Output: Dataset size: 3662
```

---

### \_\_len\_\_

Return the total number of samples in the dataset.

**Signature:**
```python
def __len__(self) -> int
```

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of samples in the dataset |

**Example:**

```python
dataset = RetinalDataset('train.csv', 'images/')
print(f"Total samples: {len(dataset)}")
# Output: Total samples: 3662
```

---

### \_\_getitem\_\_

Load and return a sample from the dataset at the given index.

**Signature:**
```python
def __getitem__(self, idx: int) -> Tuple[Image.Image, int]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `idx` | `int` | Required | Index of sample to retrieve (0 to len(dataset)-1) |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[Image.Image, int]` | Tuple of (image, label) where image is PIL Image or transformed tensor, and label is diagnosis level (0-4) |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `IndexError` | If idx is out of range [0, len(dataset)-1] |
| `FileNotFoundError` | If image file cannot be found with any valid extension |
| `IOError` | If image file cannot be loaded (corrupted image) |

**Transform Handling:**

The method automatically detects and handles two types of transforms:

1. **Albumentations** (detected by module name):
   - Converts PIL Image to numpy array
   - Applies transform with `transform(image=image_np)`
   - Returns transformed tensor

2. **Torchvision** (default):
   - Applies transform directly to PIL Image
   - Returns transformed tensor

**Example:**

```python
from scripts.dataset import RetinalDataset
from torchvision import transforms

# With torchvision transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

dataset = RetinalDataset(
    csv_file='train.csv',
    img_dir='images/',
    transform=transform
)

# Get a sample
image, label = dataset[0]
print(f"Image shape: {image.shape}")  # torch.Size([3, 224, 224])
print(f"Label: {label}")               # 0 (No DR)
```

**Albumentations Example:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# With albumentations
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

dataset = RetinalDataset(
    csv_file='train.csv',
    img_dir='images/',
    transform=transform
)

image, label = dataset[0]
print(f"Image shape: {image.shape}")  # torch.Size([3, 224, 224])
```

---

### get_class_distribution

Get the distribution of diagnosis classes in the dataset.

**Signature:**
```python
def get_class_distribution(self) -> Dict[int, int]
```

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[int, int]` | Dictionary mapping diagnosis level (0-4) to sample count |

**Example:**

```python
dataset = RetinalDataset('train.csv', 'images/')
distribution = dataset.get_class_distribution()

print("Class distribution:")
severity_names = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}

for diagnosis, count in sorted(distribution.items()):
    pct = 100.0 * count / len(dataset)
    print(f"  Class {diagnosis} ({severity_names[diagnosis]:15s}): "
          f"{count:4d} ({pct:5.1f}%)")

# Output:
#   Class 0 (No DR         ): 1805 (49.3%)
#   Class 1 (Mild NPDR     ):  370 (10.1%)
#   Class 2 (Moderate NPDR ):  999 (27.3%)
#   Class 3 (Severe NPDR   ):  193 ( 5.3%)
#   Class 4 (PDR           ):  295 ( 8.1%)
```

---

### get_sample_info

Get metadata for a specific sample without loading the image.

**Signature:**
```python
def get_sample_info(self, idx: int) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `idx` | `int` | Required | Index of the sample (0 to len(dataset)-1) |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Dictionary with keys: `id_code` (str), `diagnosis` (int), `index` (int) |

**Example:**

```python
dataset = RetinalDataset('train.csv', 'images/')

# Get metadata without loading image (fast)
info = dataset.get_sample_info(0)
print(f"Sample 0 metadata: {info}")
# Output: {'id_code': 'image_001', 'diagnosis': 0, 'index': 0}

# Useful for debugging or inspection
for i in range(5):
    info = dataset.get_sample_info(i)
    print(f"Sample {i}: ID={info['id_code']}, Diagnosis={info['diagnosis']}")
```

---

## Usage Examples

### Example 1: Basic Usage Without Transforms

```python
from scripts.dataset import RetinalDataset

# Create dataset
dataset = RetinalDataset(
    csv_file='data/aptos/train.csv',
    img_dir='data/aptos/train_images'
)

# Get dataset info
print(f"Total samples: {len(dataset)}")
print(f"Class distribution: {dataset.get_class_distribution()}")

# Load a sample
image, label = dataset[0]
print(f"Image type: {type(image)}")    # <class 'PIL.Image.Image'>
print(f"Image size: {image.size}")      # (2048, 1536)
print(f"Label: {label}")                 # 0
```

### Example 2: With Torchvision Transforms

```python
from scripts.dataset import RetinalDataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create dataset with transforms
dataset = RetinalDataset(
    csv_file='data/aptos/train.csv',
    img_dir='data/aptos/train_images',
    transform=transform
)

# Load transformed sample
image, label = dataset[0]
print(f"Image type: {type(image)}")      # <class 'torch.Tensor'>
print(f"Image shape: {image.shape}")      # torch.Size([3, 224, 224])
print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")
```

### Example 3: With Albumentations (Advanced Augmentation)

```python
from scripts.dataset import RetinalDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define advanced augmentation pipeline
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=45,
        p=0.5
    ),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ], p=0.5),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        p=0.3
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Create dataset
dataset = RetinalDataset(
    csv_file='data/aptos/train.csv',
    img_dir='data/aptos/train_images',
    transform=train_transform
)

image, label = dataset[0]
print(f"Augmented image shape: {image.shape}")  # torch.Size([3, 224, 224])
```

### Example 4: Analyzing Class Imbalance

```python
from scripts.dataset import RetinalDataset
import matplotlib.pyplot as plt

dataset = RetinalDataset('train.csv', 'images/')

# Get class distribution
distribution = dataset.get_class_distribution()

# Visualize
classes = list(distribution.keys())
counts = list(distribution.values())

plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue', edgecolor='black')
plt.xlabel('Diagnosis Level')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Set')
plt.xticks(classes, ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'])
plt.grid(axis='y', alpha=0.3)
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate class weights for balanced loss
total = sum(counts)
class_weights = [total / (len(classes) * count) for count in counts]
print(f"Class weights: {class_weights}")
```

---

## Integration with DataLoader

### Basic DataLoader Integration

```python
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Create dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = RetinalDataset(
    csv_file='train.csv',
    img_dir='images/',
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For CUDA performance
)

# Iterate over batches
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: images={images.shape}, labels={labels.shape}")
    # images: torch.Size([32, 3, 224, 224])
    # labels: torch.Size([32])
    break
```

### Train/Val Split with DataLoader

```python
from scripts.dataset import RetinalDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch

# Create full dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = RetinalDataset(
    csv_file='train.csv',
    img_dir='images/',
    transform=transform
)

# Split into train/val (80/20)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

# Create loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## Best Practices

### 1. CSV File Validation

Always validate your CSV file before creating the dataset:

```python
import pandas as pd

# Load and inspect CSV
df = pd.read_csv('train.csv')
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Check for required columns
assert 'id_code' in df.columns, "Missing 'id_code' column"
assert 'diagnosis' in df.columns, "Missing 'diagnosis' column"

# Check diagnosis range
assert df['diagnosis'].min() >= 0, "Diagnosis values must be >= 0"
assert df['diagnosis'].max() <= 4, "Diagnosis values must be <= 4"

print("✓ CSV validation passed")
```

### 2. Image Directory Structure

Organize images consistently:

```
data/
├── aptos/
│   ├── train.csv
│   ├── train_images/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── test_images/
│       └── ...
```

### 3. Transform Best Practices

```python
# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 4. Handling Class Imbalance

```python
from torch.utils.data import WeightedRandomSampler

dataset = RetinalDataset('train.csv', 'images/', transform=transform)

# Calculate sample weights
distribution = dataset.get_class_distribution()
class_weights = {cls: 1.0 / count for cls, count in distribution.items()}

# Create sample weights
sample_weights = [class_weights[dataset.data_frame.loc[i, 'diagnosis']]
                  for i in range(len(dataset))]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use with DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Don't use shuffle=True with sampler
    num_workers=4
)
```

---

## Troubleshooting

### Issue 1: FileNotFoundError for Images

**Error:**
```
FileNotFoundError: Image not found for id_code 'image_001'. Tried extensions: ['.png', '.jpg', '.jpeg']
```

**Solution:**
- Verify image file exists in `img_dir`
- Check that filename matches `id_code` in CSV (case-sensitive)
- Ensure file has a supported extension

```python
# Debug script to find missing images
import pandas as pd
from pathlib import Path

csv_file = 'train.csv'
img_dir = Path('images/')
df = pd.read_csv(csv_file)

missing = []
for id_code in df['id_code']:
    found = any((img_dir / f"{id_code}{ext}").exists()
                for ext in ['.png', '.jpg', '.jpeg'])
    if not found:
        missing.append(id_code)

print(f"Missing images: {len(missing)}")
if missing[:5]:
    print(f"Examples: {missing[:5]}")
```

### Issue 2: ValueError for Missing Columns

**Error:**
```
ValueError: CSV missing required columns: ['diagnosis']. Found columns: ['id_code', 'label']
```

**Solution:**
Rename columns in CSV to match expected names:

```python
import pandas as pd

df = pd.read_csv('train.csv')
df = df.rename(columns={'label': 'diagnosis'})
df.to_csv('train_fixed.csv', index=False)
```

### Issue 3: Invalid Diagnosis Values

**Error:**
```
ValueError: Found 5 invalid diagnosis values. Diagnosis must be in range [0, 4].
```

**Solution:**
Clean and remap diagnosis values:

```python
import pandas as pd

df = pd.read_csv('train.csv')

# Check invalid values
invalid = df[(df['diagnosis'] < 0) | (df['diagnosis'] > 4)]
print(f"Invalid rows:\n{invalid}")

# Option 1: Remove invalid rows
df = df[(df['diagnosis'] >= 0) & (df['diagnosis'] <= 4)]

# Option 2: Remap values (if using different encoding)
# df['diagnosis'] = df['diagnosis'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4})

df.to_csv('train_cleaned.csv', index=False)
```

### Issue 4: Transform Not Applied

**Issue:** Images returned as PIL Images instead of tensors

**Solution:**
Make sure to use `ToTensor()` or `ToTensorV2()`:

```python
# Torchvision
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # ← Don't forget this!
])

# Albumentations
transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()  # ← Don't forget this!
])
```

---

## See Also

- [Models API Documentation](models_api.md) - Model architectures
- [Training API Documentation](training_api.md) - Training workflows
- [Utils API Documentation](utils_api.md) - Utility functions including transforms

---

**Generated with Claude Code** | Last Updated: 2024
