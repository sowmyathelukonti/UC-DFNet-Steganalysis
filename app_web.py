import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
import threading
import time
import sys
import webbrowser
import base64

# Import local modules
from models.ucdfnet import UCDFNet
from test import GradCAM, preprocess_image, get_heatmap_overlay, run_inference, visualize_features
from utils import embed_lsb, extract_lsb, embed_lsb_matching, embed_random_path, extract_random_path, embed_dct, extract_dct, calculate_lsb_transition_rate, check_for_hidden_message, check_for_hidden_message_sequential, check_for_hidden_message_dct
from dataset.stego_dataset import generate_synthetic_dataset
from train import train_model

app = Flask(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"
model = None

def load_global_model():
    global model
    if os.path.exists(model_path):
        try:
            model = UCDFNet(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"UC-DFNet model loaded successfully onto {device}.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print("Warning: 'best_model.pth' not found. Please ensure pre-trained model weights are placed in the root directory.")

# Automatically load model on module import (Vercel serverless startup)
load_global_model()




@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/device', methods=['GET'])
def get_device():
    return jsonify({"device": str(device).upper()})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not model:
        return jsonify({"success": False, "error": "Model weights ('best_model.pth') not loaded. Please train the model in the Training Wizard first."}), 400
        
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
        
    # Ensure static folder exists
    os.makedirs('static', exist_ok=True)
    
    try:
        # Decode the uploaded file directly from memory (no disk I/O, no Windows file locking bugs)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if cv_img is None:
            return jsonify({"success": False, "error": "Failed to decode uploaded image"}), 400
            
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 1. Preprocess and Run Inference
        image_tensor, original_rgb = preprocess_image(cv_img_rgb)
        pred_label, confidence, probs = run_inference(model, image_tensor, device)
        
        is_active_stego = False
        active_details = None
        algo_type = ""
        
        if cv_img is not None:
            # Test key 42 and seeds from 1 to 100
            test_seeds = [42] + list(range(1, 101))
            res_rand = check_for_hidden_message(cv_img, keys=test_seeds)
            if res_rand.get("detected"):
                is_active_stego = True
                active_details = res_rand
                active_details["type"] = "Random Path LSB"
                algo_type = "Random Path LSB"
                
            if not is_active_stego:
                res_seq = check_for_hidden_message_sequential(cv_img)
                if res_seq.get("detected"):
                    is_active_stego = True
                    active_details = res_seq
                    active_details["type"] = "Sequential LSB/LSB Matching"
                    algo_type = "Sequential LSB/LSB Matching"
                    
            if not is_active_stego:
                res_dct = check_for_hidden_message_dct(cv_img)
                if res_dct.get("detected"):
                    is_active_stego = True
                    active_details = res_dct
                    active_details["type"] = "DCT Domain (JPEG)"
                    algo_type = "DCT Domain (JPEG)"
                    
            if is_active_stego:
                pred_label = f"Stego Image Detected ({algo_type})"
                confidence = 0.9999
        
        # 2. Run Explainability
        if is_active_stego and active_details is not None:
            h, w, _ = cv_img.shape
            cam_np = np.zeros((h, w), dtype=np.float32)
            
            if active_details["type"] == "Random Path LSB":
                seed = active_details["seed"]
                channels = active_details["channels"]
                bit_count = (active_details["charLength"] + 1) * 8
                pixel_count = int(np.ceil(bit_count / len(channels)))
                
                total_pixels = h * w
                indices = {}
                state = seed
                walk = []
                for i in range(total_pixels - 1, total_pixels - 1 - pixel_count, -1):
                    state = (1664525 * state + 1013904223) % 4294967296
                    j = state % (i + 1)
                    val_i = indices.get(i, i)
                    val_j = indices.get(j, j)
                    indices[i] = val_j
                    indices[j] = val_i
                    walk.append(val_i)
                    
                for idx in walk:
                    px_y = idx // w
                    px_x = idx % w
                    cam_np[px_y, px_x] = 1.0
                    
            elif active_details["type"] == "Sequential LSB/LSB Matching":
                channels = active_details["channels"]
                bit_count = (active_details["charLength"] + 1) * 8
                pixel_count = int(np.ceil(bit_count / len(channels)))
                
                for idx in range(pixel_count):
                    px_y = idx // w
                    px_x = idx % w
                    cam_np[px_y, px_x] = 1.0
                    
            elif active_details["type"] == "DCT Domain (JPEG)":
                channels = active_details["channels"]
                bit_count = (active_details["charLength"] + 1) * 8
                coeff_per_block = 1
                total_dct_coeffs_needed = bit_count
                coeffs_found = 0
                
                for y_start in range(0, h - 7, 8):
                    for x_start in range(0, w - 7, 8):
                        for c in channels:
                            if coeffs_found >= total_dct_coeffs_needed:
                                break
                            cam_np[y_start:y_start+8, x_start:x_start+8] = 1.0
                            coeffs_found += coeff_per_block
                        if coeffs_found >= total_dct_coeffs_needed:
                            break
                    if coeffs_found >= total_dct_coeffs_needed:
                        break
            
            # Dilate the pixel mask to make individual dots clearly visible
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cam_dilated = cv2.dilate(cam_np, kernel)
            
            overlaid_cam = original_rgb.copy()
            # Draw bright red dots over the modified pixel coordinates
            overlaid_cam[cam_dilated > 0] = [255, 0, 0]
        else:
            grad_cam = GradCAM(model, model.stage1_deb)
            pred_class = np.argmax(probs)
            cam_np, _, _ = grad_cam.generate_cam(image_tensor.to(device), target_class=pred_class)
            
            # Convert neural Grad-CAM activations to matching sparse red dots
            h_orig, w_orig, _ = original_rgb.shape
            cam_resized = cv2.resize(cam_np, (w_orig, h_orig))
            cam_max = cam_resized.max()
            if cam_max > 0:
                cam_resized = cam_resized / cam_max
                
            # Exclude the outer 4% of margins to eliminate neural padding/boundary artifacts
            border_y = max(4, int(h_orig * 0.04))
            border_x = max(4, int(w_orig * 0.04))
            inner_mask = np.zeros((h_orig, w_orig), dtype=bool)
            inner_mask[border_y:-border_y, border_x:-border_x] = True
            
            # Use a dynamic threshold based on the top 10% (90th percentile) of inner activations
            # This ensures we always show suspicious regions regardless of the absolute scale
            inner_vals = cam_resized[inner_mask]
            thresh = np.percentile(inner_vals, 90.0) if inner_vals.size > 0 else 0.5
            
            high_act = cam_resized > thresh
            grid_y, grid_x = np.mgrid[0:h_orig, 0:w_orig]
            sparse_mask = (grid_y % 3 == 0) & (grid_x % 3 == 0)
            
            dot_mask = high_act & sparse_mask & inner_mask
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cam_dilated = cv2.dilate(dot_mask.astype(np.uint8), kernel)
            
            overlaid_cam = original_rgb.copy()
            overlaid_cam[cam_dilated > 0] = [255, 0, 0]
            grad_cam.remove_hooks()
        
        # 3. Extract Intermediate Feature maps
        features_grid = visualize_features(model, image_tensor, device, layer_name="stage1_deb")
        
        # Convert output images to base64 data URLs (100% serverless compatible, 0 disk write)
        def to_b64(rgb_arr):
            _, buf = cv2.imencode('.png', cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
            return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')
            
        gradcam_b64 = to_b64(overlaid_cam)
        features_b64 = to_b64(features_grid)
        
        # Try writing to static folder for local filesystem compatibility
        try:
            os.makedirs('static', exist_ok=True)
            cv2.imwrite(os.path.join('static', 'gradcam.png'), cv2.cvtColor(overlaid_cam, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join('static', 'features.png'), cv2.cvtColor(features_grid, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
            
        return jsonify({
            "success": True,
            "prediction": pred_label,
            "confidence": confidence,
            "gradcam_url": gradcam_b64,
            "features_url": features_b64
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Error running steganalysis: {str(e)}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/embed', methods=['POST'])
def api_embed():
    if 'file' not in request.files or 'message' not in request.form:
        return jsonify({"success": False, "error": "Missing image file or text message payload"}), 400
        
    file = request.files['file']
    message = request.form['message']
    algo = request.form.get('algo', 'LSB Replacement')
    channel_desc = request.form.get('channels', 'All Channels (RGB)')
    
    try:
        param = float(request.form.get('param', '42'))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid parameter/key value"}), 400
        
    # Read uploaded image using OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Failed to decode uploaded image"}), 400
        
    # Map channels
    channel_map = {
        "All Channels (RGB)": [0, 1, 2],
        "Red Channel Only": [0],
        "Green Channel Only": [1],
        "Blue Channel Only": [2]
    }
    ch_list = channel_map.get(channel_desc, [0, 1, 2])
    
    try:
        # Run steganographic embedding
        if algo == "LSB Replacement":
            stego = embed_lsb(img, message, channels=ch_list)
        elif algo == "LSB Matching":
            stego = embed_lsb_matching(img, message, channels=ch_list)
        elif algo == "Random Path LSB":
            stego = embed_random_path(img, message, key=int(param), channels=ch_list)
        else:  # DCT Domain
            stego = embed_dct(img, message, channels=ch_list, Q=float(param))
            
        os.makedirs('static', exist_ok=True)
        stego_path = os.path.join('static', 'stego.png')
        cv2.imwrite(stego_path, stego)
        
        timestamp = int(time.time())
        return jsonify({
            "success": True,
            "download_url": f"/static/stego.png?t={timestamp}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Embedding failed: {str(e)}"}), 500

@app.route('/api/extract', methods=['POST'])
def api_extract():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Missing stego image file"}), 400
        
    file = request.files['file']
    algo = request.form.get('algo', 'LSB Replacement / Matching')
    channel_desc = request.form.get('channels', 'All Channels (RGB)')
    
    try:
        param = float(request.form.get('param', '42'))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid parameter/key value"}), 400
        
    # Decode image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Failed to decode uploaded stego image"}), 400
        
    # Map channels
    channel_map = {
        "All Channels (RGB)": [0, 1, 2],
        "Red Channel Only": [0],
        "Green Channel Only": [1],
        "Blue Channel Only": [2]
    }
    ch_list = channel_map.get(channel_desc, [0, 1, 2])
    
    try:
        if algo == "LSB Replacement / Matching":
            extracted = extract_lsb(img, channels=ch_list)
        elif algo == "Random Path LSB":
            extracted = extract_random_path(img, key=int(param), channels=ch_list)
        else:  # DCT
            extracted = extract_dct(img, channels=ch_list, Q=float(param))
            
        return jsonify({
            "success": True,
            "message": extracted
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Extraction failed: {str(e)}"}), 500



@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def open_browser():
    try:
        # Open browser to local Flask instance
        webbrowser.open_new("http://127.0.0.1:5000")
    except Exception as e:
        print(f"Failed to open browser automatically: {e}")

if __name__ == '__main__':
    # Load model weights on startup
    load_global_model()
    
    # Automatically open the browser after a 1.2 second delay (once Flask starts)
    threading.Timer(1.2, open_browser).start()
    
    # Run flask local server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
