import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import roi_pool
import os

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
        
        # Output layers
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
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
        
        # Classification and bbox regression outputs
        cls_scores = self.cls_score(out)
        bbox_deltas = self.bbox_pred(out)
        
        return cls_scores, bbox_deltas


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21  # 20 object classes + 1 background
    
    model = FastRCNN_VGG16(num_classes=num_classes).to(device)
    model.train()
    
    # Define optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss(reduction='sum')
    
    # Create dummy data
    dummy_image = torch.randn(2, 3, 224, 224).to(device)
    
    # 4 ROIs: [batch_idx, x1, y1, x2, y2]
    dummy_rois = torch.tensor([
        [0, 10, 10, 50, 50],
        [0, 30, 30, 100, 100],
        [1, 20, 20, 60, 60],
        [1, 50, 50, 120, 120]
    ], dtype=torch.float).to(device)
    
    # Ground truth labels (0 is background, >0 are object classes)
    gt_labels = torch.tensor([5, 0, 2, 20], dtype=torch.long).to(device)
    
    # Ground truth bbox regression targets [dx, dy, dw, dh]
    gt_bbox_targets = torch.randn(4, 4).to(device)
    
    # Original ROI locations (before regression)
    roi_locations = dummy_rois[:, 1:5].clone()  # [x1, y1, x2, y2] for each ROI
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    cls_scores, bbox_pred = model(dummy_image, dummy_rois)
    
    # Calculate classification loss
    loss_cls = criterion_cls(cls_scores, gt_labels)
    
    # Calculate regression loss (only for foreground ROIs)
    fg_mask = gt_labels > 0
    fg_indices = torch.nonzero(fg_mask).squeeze()
    loss_loc = torch.tensor(0.0).to(device)
    
    if fg_indices.numel() > 0:
        bbox_pred_reshaped = bbox_pred.view(bbox_pred.size(0), num_classes, 4)
        fg_labels = gt_labels[fg_mask]
        pred_locs = bbox_pred_reshaped[fg_mask, fg_labels, :]
        target_locs = gt_bbox_targets[fg_mask]
        
        loss_loc = criterion_reg(pred_locs, target_locs)
        loss_loc = loss_loc / (gt_labels.numel() + 1e-5)
    
    # Total loss and backward
    total_loss = loss_cls + loss_loc
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
        f.write(f"Classification Loss: {loss_cls.item():.4f}\n")
        f.write(f"Regression Loss (Smooth L1): {loss_loc.item():.4f}\n")
        f.write(f"Total Loss: {total_loss.item():.4f}\n")
        
        # Test: Compute IoU > 0.5 for regression evaluation
        f.write("\n" + "="*60 + "\n")
        f.write("Bounding Box Regression Evaluation (IoU > 0.5)\n")
        f.write("="*60 + "\n")
    
        # Apply predicted deltas to original ROIs to get predicted boxes
        predicted_boxes = []
        bbox_pred_reshaped = bbox_pred.view(bbox_pred.size(0), num_classes, 4)
        
        for i in range(len(gt_labels)):
            roi_box = roi_locations[i].cpu().detach().numpy()
            pred_class = torch.argmax(cls_scores[i]).item()
            
            # Get deltas for predicted class
            deltas = bbox_pred_reshaped[i, pred_class, :].cpu().detach().numpy()
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
        f.write(f"\n{'ROI':<5} {'GT Class':<15} {'GT Box':<30} {'Pred Box':<30} {'IoU':<10}\n")
        f.write("-" * 100 + "\n")
        
        high_iou_count = 0
        for i in range(len(gt_labels)):
            roi_box = roi_locations[i].cpu().detach().numpy()
            gt_class = gt_labels[i].item()
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
            
            f.write(f"{i:<5} {gt_class:<15} {roi_box_str:<30} {pred_box_str:<30} {iou:<10.4f} {status}\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"\nSummary: {high_iou_count}/{len(gt_labels)} ROIs achieved IoU > 0.5\n")
        f.write(f"Success Rate: {100 * high_iou_count / len(gt_labels):.1f}%\n")
    
    print(f"Results saved to {output_file}")