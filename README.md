# Fast R-CNN Implementation

A PyTorch implementation of Fast R-CNN for object detection using VGG16 backbone with Smooth L1 loss for bounding box regression.

## Project Overview

This project implements the Fast R-CNN architecture as described in the original paper, featuring:

- **VGG16 Backbone**: Pre-configured 16-layer convolutional neural network for feature extraction
- **ROI Pooling**: Region of Interest pooling layer to extract fixed-size features from variable-sized regions
- **Bounding Box Regression**: Smooth L1 loss for accurate localization with learnable regression targets
- **Fully Connected Layers**: Multi-layer perceptron for classification and bounding box prediction

## Architecture Details

### VGG16 Feature Extractor
- 5 convolutional blocks with ReLU activations
- Max pooling after each block
- Output: 512-channel feature maps

### ROI Pooling
- Spatial scale: 1/16 (matches VGG16 stride)
- Output pooled size: 7×7 features per ROI

### Head Network
- FC6: 512 × 7 × 7 → 4096 features with dropout (p=0.5)
- FC7: 4096 → 4096 features with dropout (p=0.5)
- Output: 4 bounding box regression parameters (dx, dy, dw, dh)

## Loss Function

**Smooth L1 Loss** for bounding box regression:
- Combines quadratic loss for small errors and linear loss for large errors
- More robust to outliers compared to standard L2 loss
- Default beta threshold: 1.0 (customizable)

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, CPU supported)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Fast-RCnn
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 matplotlib
```

## Usage

### Running the Demo

```bash
cd "Simple CNN"
python test.py
```

This will:
1. Initialize FastRCNN_VGG16 model
2. Create dummy image and ROI data with noise
3. Perform forward pass through the network
4. Calculate Smooth L1 regression loss
5. Compute IoU metrics for predicted boxes
6. Generate visualizations of predictions
7. Save results to `out/results.txt`

### Output Files

- `out/results.txt`: Detailed loss metrics and IoU evaluation
- `out/roi_*.png`: Visualization of ground truth vs predicted bounding boxes for each ROI

## Key Components

### FastRCNN_VGG16 Class

```python
model = FastRCNN_VGG16(num_classes=21)
bbox_deltas = model(image, rois)
```

**Parameters:**
- `num_classes`: Number of object classes (default: 21)

**Inputs:**
- `image`: Batch of images (B, 3, H, W)
- `rois`: Region proposals (N, 5) where each ROI is [batch_idx, x1, y1, x2, y2]

**Outputs:**
- `bbox_deltas`: Bounding box regression parameters (N, 4)

### Loss Computation

```python
criterion_reg = nn.SmoothL1Loss(reduction='sum', beta=1.0)
loss = criterion_reg(predicted_boxes, ground_truth_boxes)
```

**Customize Beta Threshold:**
```python
# Lower beta for more aggressive loss transition
criterion_reg = nn.SmoothL1Loss(reduction='sum', beta=0.5)

# Higher beta for smoother transition
criterion_reg = nn.SmoothL1Loss(reduction='sum', beta=2.0)
```

### Evaluation Metrics

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth boxes
- Success criterion: IoU > 0.5 per region

## File Structure

```
Fast-RCnn/
├── README.md
├── requirements.txt
├── Simple CNN/
│   ├── test.py              # Main implementation and demo
│   └── out/                 # Output directory (created at runtime)
│       ├── results.txt
│       └── roi_*.png
```

## Configuration Parameters

Edit `test.py` to customize:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Learning Rate | line 238 | 0.001 | Optimizer learning rate |
| Momentum | line 238 | 0.9 | SGD momentum |
| Beta (Smooth L1) | line 239 | 1.0 | Smooth L1 loss threshold |
| Dropout Rate | lines 62, 75 | 0.5 | FC layer dropout probability |
| ROI Pooled Size | line 62 | (7, 7) | Output size after ROI pooling |

## Performance Notes

- Current implementation uses dummy data for demonstration
- For production use, integrate with real dataset loading pipeline
- Adjust beta threshold based on regression target scale
- Monitor loss values to detect training issues

## Future Enhancements

- [ ] Add classification head
- [ ] Implement Faster R-CNN with RPN
- [ ] Add multi-scale training
- [ ] Integrate with popular datasets (PASCAL VOC, COCO)
- [ ] Add batch normalization
- [ ] Optimize for inference speed

## References

- Girshick, R. (2015). Fast R-CNN. *ICCV*
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*
