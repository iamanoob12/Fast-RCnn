import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import roi_pool
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


class FastRCNN_VGG16(nn.Module):
    def __init__(self, num_classes=21):
        super(FastRCNN_VGG16, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        
        # ROI pooling parameters
        self.spatial_scale = 1.0 / 16
        self.pooled_size = (7, 7)
        flat_dim = 512 * 7 * 7
        
        # Fully connected layers
        self.fc6_L = nn.Linear(flat_dim, 1024, bias=False)
        self.fc6_U = nn.Linear(1024, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.5)
        
        self.fc7_L = nn.Linear(4096, 256, bias=False)
        self.fc7_U = nn.Linear(256, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=0.5)
        
        # Output layer - only bbox regression
        self.bbox_pred = nn.Linear(4096, 4)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x, rois):
        # VGG16 backbone
        # Block 1
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        
        # ROI pooling
        pool5 = roi_pool(x, rois, self.pooled_size, self.spatial_scale)
        out = pool5.view(pool5.size(0), -1)
        
        # FC6
        out = self.fc6_L(out)
        out = self.fc6_U(out)
        out = self.relu6(out)
        out = self.drop6(out)
        
        # FC7
        out = self.fc7_L(out)
        out = self.fc7_U(out)
        out = self.relu7(out)
        out = self.drop7(out)
        
        # Bbox regression output only
        bbox_deltas = self.bbox_pred(out)
        
        return bbox_deltas


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-5)
    return iou


def visualize_predictions(image, roi_locations, predicted_boxes, output_dir, filename="predictions.png"):
    """Visualize each ROI and its predicted box in separate images"""
    # Convert image tensor to numpy for visualization
    img_np = image[0].cpu().detach().numpy()  # Take first image from batch
    
    # Normalize image to [0, 1] if needed
    if img_np.max() > 1:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    
    # Convert from CHW to HWC for matplotlib
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    output_paths = []
    
    # Create separate image for each ROI
    for i in range(len(roi_locations)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_np, cmap='gray' if img_np.shape[2] == 1 else None)
        
        # Draw this ROI in solid green
        roi = roi_locations[i]
        x1, y1, x2, y2 = roi.cpu().detach().numpy()
        width = x2 - x1
        height = y2 - y1
        rect_roi = patches.Rectangle((x1, y1), width, height, linewidth=3, 
                                      edgecolor='lime', facecolor='none', label='Ground Truth ROI')
        ax.add_patch(rect_roi)
        # Place label inside box with background
        ax.text(x1 + 5, y1 + 15, f'GT{i}', color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lime', alpha=0.8, edgecolor='white', linewidth=2))
        
        # Draw corresponding predicted box with dashed line
        pred_box = predicted_boxes[i]
        x1_p, y1_p, x2_p, y2_p = pred_box
        width_p = x2_p - x1_p
        height_p = y2_p - y1_p
        rect_pred = patches.Rectangle((x1_p, y1_p), width_p, height_p, linewidth=3, 
                                       edgecolor='cyan', facecolor='none', linestyle='--',
                                       label='Predicted Box')
        ax.add_patch(rect_pred)
        # Place label inside box with background
        ax.text(x1_p + 5, y2_p - 5, f'P{i}', color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='cyan', alpha=0.8, edgecolor='white', linewidth=2))
        
        ax.set_title(f'ROI {i}: Ground Truth vs Predicted Bounding Box', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.axis('on')
        
        # Save individual figure
        filename_individual = f"roi_{i}_prediction.png"
        output_path = os.path.join(output_dir, filename_individual)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_paths.append(output_path)
    
    return output_paths


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FastRCNN_VGG16().to(device)
    model.train()
    
    # Define optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion_reg = nn.SmoothL1Loss(reduction='sum')
    
    # Create dummy data with EXTREME noise
    dummy_image = torch.randn(2, 3, 224, 224).to(device) * 10.0  # Extreme magnitude
    # Add multiple layers of extreme Gaussian noise
    noise1 = torch.randn_like(dummy_image) * 5.0  # Heavy layer 1
    noise2 = torch.randn_like(dummy_image) * 3.0  # Heavy layer 2
    dummy_image = dummy_image + noise1 + noise2
    # Add heavy salt-and-pepper artifacts (50% of pixels)
    random_mask1 = torch.rand_like(dummy_image) > 0.5
    dummy_image[random_mask1] = torch.randn_like(dummy_image[random_mask1]) * 20.0
    # Add uniform random noise
    uniform_noise = torch.rand_like(dummy_image) * 15.0
    dummy_image = dummy_image + uniform_noise
    
    # 4 ROIs with EXTREME noise: [batch_idx, x1, y1, x2, y2]
    roi_noise = torch.randn(4, 4) * 30  # Extreme noise to ROI coordinates
    roi_noise2 = torch.randn(4, 4) * 20  # Extra noise layer
    dummy_rois = torch.tensor([
        [0, 10, 10, 50, 50],
        [0, 30, 30, 100, 100],
        [1, 20, 20, 60, 60],
        [1, 50, 50, 120, 120]
    ], dtype=torch.float)
    dummy_rois[:, 1:5] = dummy_rois[:, 1:5] + roi_noise + roi_noise2  # Add extreme noise to coordinates
    dummy_rois = dummy_rois.to(device)
    
    # Ground truth bbox regression targets with EXTREME noise [dx, dy, dw, dh]
    gt_bbox_targets = torch.randn(4, 4).to(device) * 10.0  # Extreme noise magnitude
    
    # Original ROI locations (before regression)
    roi_locations = dummy_rois[:, 1:5].clone()  # [x1, y1, x2, y2] for each ROI
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    bbox_pred = model(dummy_image, dummy_rois)
    
    # Calculate regression loss
    loss_loc = criterion_reg(bbox_pred, gt_bbox_targets)
    loss_loc = loss_loc / (bbox_pred.size(0) + 1e-5)
    
    # Total loss and backward
    total_loss = loss_loc
    total_loss.backward()
    optimizer.step()
    
    # Create output directory in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "out")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open output file with UTF-8 encoding
    output_file = os.path.join(output_dir, "results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Regression Loss (Smooth L1): {loss_loc.item():.4f}\n")
        
        # Test: Compute IoU > 0.5 for regression evaluation
        f.write("\n" + "="*60 + "\n")
        f.write("Bounding Box Regression Evaluation (IoU > 0.5)\n")
        f.write("="*60 + "\n")
    
        # Apply predicted deltas to original ROIs to get predicted boxes
        predicted_boxes = []
        
        for i in range(bbox_pred.size(0)):
            roi_box = roi_locations[i].cpu().detach().numpy()
            
            # Get deltas
            deltas = bbox_pred[i, :].cpu().detach().numpy()
            dx, dy, dw, dh = deltas
            
            # Apply deltas to ROI (simple transformation)
            x1, y1, x2, y2 = roi_box
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            # Apply regression deltas
            pred_cx = cx + dx * w
            pred_cy = cy + dy * h
            pred_w = w * (2 ** dw)
            pred_h = h * (2 ** dh)
            
            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2
            
            predicted_boxes.append([pred_x1, pred_y1, pred_x2, pred_y2])
        
        # Compute IoU with ground truth boxes
        f.write(f"\n{'ROI':<5} {'GT Box':<30} {'Pred Box':<30} {'IoU':<10}\n")
        f.write("-" * 100 + "\n")
        
        high_iou_count = 0
        for i in range(len(predicted_boxes)):
            roi_box = roi_locations[i].cpu().detach().numpy()
            pred_box = predicted_boxes[i]
            
            iou = compute_iou(roi_box, pred_box)
            
            # Check if IoU > 0.5
            if iou > 0.5:
                high_iou_count += 1
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            
            roi_box_str = f"({roi_box[0]:.1f}, {roi_box[1]:.1f}, {roi_box[2]:.1f}, {roi_box[3]:.1f})"
            pred_box_str = f"({pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f})"
            
            f.write(f"{i:<5} {roi_box_str:<30} {pred_box_str:<30} {iou:<10.4f} {status}\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"\nSummary: {high_iou_count}/{len(predicted_boxes)} ROIs achieved IoU > 0.5\n")
        f.write(f"Success Rate: {100 * high_iou_count / len(predicted_boxes):.1f}%\n")
    
    print(f"Results saved to {output_file}")
    
    # Visualize predictions - creates separate image for each ROI
    visualization_paths = visualize_predictions(dummy_image, roi_locations, predicted_boxes, output_dir)
    print(f"Visualizations saved:")
    for path in visualization_paths:
        print(f"  - {path}")