import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local modules
from models.ucdfnet import UCDFNet
from dataset.stego_dataset import get_augmentations

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for explaining 
    predictions of UC-DFNet by highlighting suspicious image regions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        
        # Compatibility with older/newer PyTorch backward hooks
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
            
    def generate_cam(self, input_tensor, target_class=None):
        """Generates heatmap matrix for a given class."""
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            # Predict the class with the highest probability
            target_class = torch.argmax(logits, dim=1).item()
            
        # Backward pass for the target class
        score = logits[0, target_class]
        score.backward()
        
        # Compute weights as the spatial average of gradients
        gradients = self.gradients  # Shape: (1, C, H, W)
        activations = self.activations  # Shape: (1, C, H, W)
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Shape: (1, 1, H, W)
        
        # Apply ReLU to retain only positive influence features
        cam = F.relu(cam)
        
        # Normalize CAM to range [0, 1]
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        # Remove dimensions to return 2D numpy array
        cam_np = cam.squeeze().cpu().numpy()
        return cam_np, target_class, logits
        
    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()


def preprocess_image(image_path_or_array):
    """
    Preprocess image for model input:
    - Resize, transform to PyTorch tensor, normalize.
    """
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path_or_array}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already an RGB numpy array
        image = image_path_or_array
        
    # Resize and apply val transforms
    val_transform = get_augmentations("val")
    pil_img = Image.fromarray(image)
    tensor_img = val_transform(pil_img).unsqueeze(0)  # Add batch dimension
    
    return tensor_img, image


def get_heatmap_overlay(original_img, cam_np, alpha=0.5):
    """
    Overlays the Grad-CAM heatmap onto the original RGB image.
    Returns:
    - superimposed_img: Heatmap overlay on original image
    - heatmap_colored: Heatmap image itself (colored RGB)
    """
    # Resize cam to match original image dimensions
    h, w, _ = original_img.shape
    cam_resized = cv2.resize(cam_np, (w, h))
    
    # Scale to 0-255 range and convert to uint8
    cam_uint8 = np.uint8(255 * cam_resized)
    
    # Apply Colormap (JET)
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 1.0 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed_img, heatmap_colored


def run_inference(model, image_tensor, device):
    """Run model inference and return predicted class, probabilities, and confidence."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_class = int(np.argmax(probs))
    class_labels = {0: "Clean Image", 1: "Stego Image Detected"}
    pred_label = class_labels[pred_class]
    confidence = float(probs[pred_class])
    
    return pred_label, confidence, probs


def visualize_features(model, image_tensor, device, layer_name="stage1_deb"):
    """
    Extracts intermediate feature maps for visualization.
    Returns a grid of feature maps (channels).
    """
    model.eval()
    with torch.no_grad():
        features_dict = model.get_stage_features(image_tensor.to(device))
        
    if layer_name not in features_dict:
        raise ValueError(f"Layer {layer_name} not available in features! Choose from: {list(features_dict.keys())}")
        
    feature_map = features_dict[layer_name].squeeze(0).cpu().numpy()  # (C, H, W)
    num_channels = feature_map.shape[0]
    
    # Take first 16 channels (or fewer if less are available)
    grid_size = min(16, num_channels)
    cols = 4
    rows = (grid_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()
    
    for i in range(grid_size):
        ax = axes[i]
        # Normalize channel to 0-1 for plotting
        ch_map = feature_map[i]
        ch_min, ch_max = ch_map.min(), ch_map.max()
        if ch_max > ch_min:
            ch_map = (ch_map - ch_min) / (ch_max - ch_min)
            
        ax.imshow(ch_map, cmap='viridis')
        ax.set_title(f"Ch {i}")
        ax.axis('off')
        
    # Turn off unused axes
    for i in range(grid_size, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # Save to a temporary image and return it as a numpy array
    temp_path = "temp_features.png"
    plt.savefig(temp_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    grid_img = cv2.imread(temp_path)
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return grid_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UC-DFNet Testing and Explainability Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to input color image")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to trained model weight checkpoint")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save explanation visualizations")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint '{args.model_path}' not found! Run train.py first.")
        exit(1)
        
    model = UCDFNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Successfully loaded model weight from '{args.model_path}'")
    
    # Preprocess image
    image_tensor, original_img = preprocess_image(args.image)
    
    # Run Inference
    pred_label, confidence, probs = run_inference(model, image_tensor, device)
    print(f"\n================== INFERENCE RESULTS ==================")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Probabilities - Clean: {probs[0]*100:.2f}% | Stego: {probs[1]*100:.2f}%")
    print("=======================================================")
    
    # Run Explainability: Grad-CAM
    print("\nGenerating Grad-CAM visualization...")
    # Target layer is the first Stage Dual-Path Enhancement Block stage1_deb
    grad_cam = GradCAM(model, model.stage1_deb)
    
    # Generate CAM for predicted class
    pred_class = np.argmax(probs)
    cam_np, _, _ = grad_cam.generate_cam(image_tensor.to(device), target_class=pred_class)
    
    # Get overlaid image
    overlaid_img, heatmap = get_heatmap_overlay(original_img, cam_np)
    
    # Save visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(args.output_dir, "original.png"), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "gradcam_heatmap.png"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, "gradcam_overlay.png"), cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    # Extract intermediate features
    print("Extracting intermediate stage 1 feature maps...")
    features_grid = visualize_features(model, image_tensor, device, layer_name="stage1_deb")
    cv2.imwrite(os.path.join(args.output_dir, "features_stage1_deb.png"), cv2.cvtColor(features_grid, cv2.COLOR_RGB2BGR))
    
    print(f"All visualizations saved successfully in '{args.output_dir}' directory!")
