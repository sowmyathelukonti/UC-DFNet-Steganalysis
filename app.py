import streamlit as st
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import time

# Import local modules
from models.ucdfnet import UCDFNet
from test import GradCAM, preprocess_image, get_heatmap_overlay, run_inference, visualize_features
from utils import embed_lsb, extract_lsb, embed_lsb_matching, embed_random_path, extract_random_path, embed_dct, extract_dct
from dataset.stego_dataset import generate_synthetic_dataset
from train import train_model

# Setup page config
st.set_page_config(
    page_title="UC-DFNet Steganalysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism, fonts, and colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF3366 0%, #FF9933 50%, #00CCFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #8A9AA6;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .status-card-clean {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .status-card-stego {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .status-title-clean {
        color: #2ECC71;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .status-title-stego {
        color: #E74C3C;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="main-title">UC-DFNet Steganalysis Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Universal Color Dual-Path Fractal Network for Color Image Steganalysis & Visual Interpretability</div>', unsafe_allow_html=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check model availability
model_path = "best_model.pth"
model_loaded = False
model = None

# Sidebar Content
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛠️ Model Control Panel</div>', unsafe_allow_html=True)
    
    if os.path.exists(model_path):
        st.success("✅ UC-DFNet model weights loaded!")
        try:
            # We initialize and cache the model
            @st.cache_resource
            def load_cached_model(path):
                net = UCDFNet(num_classes=2)
                net.load_state_dict(torch.load(path, map_location=device))
                net.to(device)
                net.eval()
                return net
            
            model = load_cached_model(model_path)
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    else:
        st.warning("⚠️ Model weights ('best_model.pth') not found.")
        
    # Model training trigger in sidebar
    with st.expander("🎓 Prototype Training Wizard", expanded=not model_loaded):
        st.write("Train a lightweight UC-DFNet model on synthetic cover/stego images to activate analysis.")
        num_samples = st.slider("Samples per class", min_value=10, max_value=200, value=50, step=10)
        epochs = st.slider("Training Epochs", min_value=1, max_value=20, value=5, step=1)
        
        if st.button("🚀 Run Synthetic Training"):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Subclass arguments to pass to training script
            class DummyArgs:
                def __init__(self):
                    self.data_dir = "data"
                    self.num_samples = num_samples
                    self.epochs = epochs
                    self.batch_size = 8
                    self.lr = 0.001
                    self.patience = 5
                    self.model_path = "best_model.pth"
            
            args = DummyArgs()
            
            with st.spinner("Training model, please wait..."):
                status_placeholder.info("Generating synthetic texture dataset...")
                # Generate dataset
                generate_synthetic_dataset(args.data_dir, num_samples=args.num_samples)
                
                status_placeholder.info("Running training epochs...")
                # Run train pipeline (it will save best_model.pth and plots)
                train_model(args)
                
            status_placeholder.success("🎉 Training complete! Model loaded.")
            st.rerun()

    # Sidebar Steganography Playground
    st.markdown('<div class="sidebar-header">🎨 Steganography Sandbox</div>', unsafe_allow_html=True)
    pg_mode = st.selectbox("Select Sandbox Operation", ["None", "Embed Message", "Extract Message"])
    
    if pg_mode == "Embed Message":
        st.write("Embed text into a clean cover image using multiple algorithms.")
        raw_img_file = st.file_uploader("Upload Clean Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="embed_upload")
        
        algo = st.selectbox("Embedding Algorithm", ["LSB Replacement", "LSB Matching", "Random Path LSB", "DCT Domain LSB"])
        
        channel_desc = st.selectbox("Channels to use", ["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"])
        channel_map = {
            "All Channels (RGB)": [0, 1, 2],
            "Red Channel Only": [0],
            "Green Channel Only": [1],
            "Blue Channel Only": [2]
        }
        ch_list = channel_map[channel_desc]
        
        key = 42
        dct_q = 16.0
        if algo == "Random Path LSB":
            key = st.number_input("Secret Key (Integer)", min_value=1, max_value=999999, value=42, key="embed_key")
        elif algo == "DCT Domain LSB":
            dct_q = st.slider("Quantization Step (Higher = More Robust but More Distorted)", min_value=4.0, max_value=64.0, value=16.0, step=4.0, key="embed_q")
            
        secret_msg = st.text_input("Enter Secret Message", "My secret stegano payload 🤫")
        
        if raw_img_file and secret_msg:
            # Read image
            file_bytes = np.asarray(bytearray(raw_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Embed message
            try:
                if algo == "LSB Replacement":
                    stego_img = embed_lsb(img, secret_msg, channels=ch_list)
                elif algo == "LSB Matching":
                    stego_img = embed_lsb_matching(img, secret_msg, channels=ch_list)
                elif algo == "Random Path LSB":
                    stego_img = embed_random_path(img, secret_msg, key=int(key), channels=ch_list)
                else:
                    stego_img = embed_dct(img, secret_msg, channels=ch_list, Q=float(dct_q))
                
                # Convert BGR to RGB for previewing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                stego_rgb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="Original Cover", width="stretch")
                with col2:
                    st.image(stego_rgb, caption="Generated Stego Image", width="stretch")
                
                # Save stego to byte array for download
                is_success, buffer = cv2.imencode(".png", stego_img)
                if is_success:
                    st.download_button(
                        label="💾 Download Stego Image",
                        data=buffer.tobytes(),
                        file_name="stego_image.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                
    elif pg_mode == "Extract Message":
        st.write("Extract hidden text embedded in an image.")
        stego_img_file = st.file_uploader("Upload Stego Image (PNG)", type=["png"], key="extract_upload")
        
        algo = st.selectbox("Expected Algorithm", ["LSB Replacement / Matching", "Random Path LSB", "DCT Domain LSB"])
        
        channel_desc = st.selectbox("Expected Channels", ["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"])
        channel_map = {
            "All Channels (RGB)": [0, 1, 2],
            "Red Channel Only": [0],
            "Green Channel Only": [1],
            "Blue Channel Only": [2]
        }
        ch_list = channel_map[channel_desc]
        
        key = 42
        dct_q = 16.0
        if algo == "Random Path LSB":
            key = st.number_input("Secret Key (Integer)", min_value=1, max_value=999999, value=42, key="extract_key")
        elif algo == "DCT Domain LSB":
            dct_q = st.slider("Quantization Step", min_value=4.0, max_value=64.0, value=16.0, step=4.0, key="extract_q")
            
        if stego_img_file:
            file_bytes = np.asarray(bytearray(stego_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if st.button("🔑 Extract Message"):
                try:
                    if algo == "LSB Replacement / Matching":
                        extracted_str = extract_lsb(img, channels=ch_list)
                    elif algo == "Random Path LSB":
                        extracted_str = extract_random_path(img, key=int(key), channels=ch_list)
                    else:
                        extracted_str = extract_dct(img, channels=ch_list, Q=float(dct_q))
                        
                    if extracted_str:
                        st.info("Extracted Payload:")
                        st.code(extracted_str)
                    else:
                        st.warning("No message detected (extracted null string).")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")


# Main Panel Content
if not model_loaded:
    st.info("💡 **Welcome to UC-DFNet!** Please use the **Prototype Training Wizard** in the sidebar on the left to generate synthetic data and train a quick model. Once training is complete, the main analysis panel will activate.")
    
    # Show theoretical info card if model not trained
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Network Architecture Highlights")
    st.markdown("""
    The **Universal Color Dual-Path Fractal Network (UC-DFNet)** is designed for color steganography detection by integrating:
    1. **Dual-Path Enhancement Block (DEB)**: Parallel residual and dense feature paths that amplify weak steganographic perturbations.
    2. **Coordinate Attention (CA)**: Highlights spatial regions that contain suspicious micro-modifications by applying direction-aware attention.
    3. **Fractal Downsampling Block (FDB)**: Multi-branch downsampling using stride-2 and stride-1 convolutions instead of standard pooling to preserve high-frequency stegano-noise.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Main Analysis Interface
    col_input, col_result = st.columns([1, 1])
    
    uploaded_image = None
    
    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Input Image Selection")
        
        src_option = st.radio("Choose image source:", ["Upload an Image", "Use a Synthetic Sample"])
        
        if src_option == "Upload an Image":
            img_file = st.file_uploader("Upload color image (PNG or JPG):", type=["png", "jpg", "jpeg"])
            if img_file is not None:
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
        else:
            # Load from synthetic dataset if available
            data_dir = "data"
            cover_dir = os.path.join(data_dir, "cover")
            stego_dir = os.path.join(data_dir, "stego")
            
            if os.path.exists(cover_dir) and os.path.exists(stego_dir):
                covers = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith('.png')]
                stegos = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith('.png')]
                
                all_samples = []
                for p in covers:
                    all_samples.append((os.path.basename(p), p, "Cover"))
                for p in stegos:
                    all_samples.append((os.path.basename(p), p, "Stego"))
                    
                selected_sample = st.selectbox(
                    "Select a synthetic image:",
                    options=range(len(all_samples)),
                    format_func=lambda idx: f"{all_samples[idx][0]} ({all_samples[idx][2]} ground truth)"
                )
                
                if len(all_samples) > 0:
                    path = all_samples[selected_sample][1]
                    uploaded_image = cv2.imread(path)
                    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
            else:
                st.info("No synthetic samples found. Run training in the sidebar first to generate data.")
                
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Selected Image Preview (Resized internally to 256x256)", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_result:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2. Steganalysis Analysis")
        
        if uploaded_image is not None:
            analyze_btn = st.button("🔍 Run UC-DFNet Steganalysis", key="analyze_btn")
            
            if analyze_btn:
                # Preprocess image
                image_tensor, original_rgb = preprocess_image(uploaded_image)
                
                # Run model prediction
                with st.spinner("Analyzing image features..."):
                    pred_label, confidence, probs = run_inference(model, image_tensor, device)
                    
                    # Store variables in session state for downstream visualizations
                    st.session_state['analyzed'] = True
                    st.session_state['pred_label'] = pred_label
                    st.session_state['confidence'] = confidence
                    st.session_state['probs'] = probs
                    st.session_state['image_tensor'] = image_tensor
                    st.session_state['original_rgb'] = original_rgb
                    
                    # Grad-CAM target layers
                    grad_cam = GradCAM(model, model.stage3_fdb)
                    pred_class = np.argmax(probs)
                    cam_np, _, _ = grad_cam.generate_cam(image_tensor.to(device), target_class=pred_class)
                    overlaid, heatmap = get_heatmap_overlay(original_rgb, cam_np)
                    grad_cam.remove_hooks()
                    
                    st.session_state['gradcam_overlay'] = overlaid
                    st.session_state['gradcam_heatmap'] = heatmap
                    
                    # Extract intermediate features
                    features_img = visualize_features(model, image_tensor, device, layer_name="stage1_deb")
                    st.session_state['features_img'] = features_img
                    
        # Check if we have results to display
        if st.session_state.get('analyzed', False):
            pred_label = st.session_state['pred_label']
            confidence = st.session_state['confidence']
            probs = st.session_state['probs']
            
            is_stego = "Stego" in pred_label
            card_class = "status-card-stego" if is_stego else "status-card-clean"
            title_class = "status-title-stego" if is_stego else "status-title-clean"
            icon = "🚨" if is_stego else "🛡️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 3rem;">{icon}</div>
                <div class="{title_class}">{pred_label.upper()}</div>
                <div class="metric-value">{confidence * 100:.2f}%</div>
                <div style="color: #8A9AA6; font-size: 0.9rem; margin-top: 5px;">Analysis Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Details Bar chart
            st.write("Class Probabilities:")
            prob_dict = {
                "Clean Cover": float(probs[0]),
                "Stego Image": float(probs[1])
            }
            st.bar_chart(prob_dict)
        else:
            st.info("Upload/select an image and click the **Run UC-DFNet Steganalysis** button to perform classification.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Explainability & Visualizations Section (displayed under columns)
    if st.session_state.get('analyzed', False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3. Explainability and Spatial Analysis (Grad-CAM)")
        st.write("Grad-CAM highlights regions where the model detected steganographic perturbations. Red areas indicate high influence, while blue/green indicate lower influence.")
        
        col_orig, col_hmap, col_over = st.columns(3)
        
        with col_orig:
            st.image(st.session_state['original_rgb'], caption="Original Color Image", width="stretch")
        with col_hmap:
            st.image(st.session_state['gradcam_heatmap'], caption="Grad-CAM Heatmap", width="stretch")
        with col_over:
            st.image(st.session_state['gradcam_overlay'], caption="Superimposed Heatmap Overlay", width="stretch")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("4. UC-DFNet Intermediate Feature Visualizations")
        st.write("Below are the activation patterns from the **Dual-Path Enhancement Block (DEB)** in the first stage of the network, highlighting how the parallel residual and dense paths extract high-frequency steganographic artifacts.")
        
        # Display the extracted features image grid
        st.image(st.session_state['features_img'], caption="Stage 1 DEB Feature Channel Activations (First 16 channels)", width="stretch")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show training metric curves if they exist
        if os.path.exists("training_curves.png"):
            with st.expander("📈 View Model Training Performance Curves"):
                st.write("These curves illustrate the training loss, validation loss, training accuracy, and validation accuracy achieved by the UC-DFNet model during training.")
                st.image("training_curves.png", caption="UC-DFNet Training & Validation Curves", width="stretch")
                
        if os.path.exists("confusion_matrix.png"):
            with st.expander("📊 View Confusion Matrix"):
                st.image("confusion_matrix.png", caption="UC-DFNet Confusion Matrix on Validation Set", width=500)
