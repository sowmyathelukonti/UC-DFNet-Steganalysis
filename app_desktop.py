import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import random

# Import local modules
from models.ucdfnet import UCDFNet
from test import GradCAM, preprocess_image, get_heatmap_overlay, run_inference, visualize_features
from utils import embed_lsb, extract_lsb, embed_lsb_matching, embed_random_path, extract_random_path, embed_dct, extract_dct
from dataset.stego_dataset import generate_synthetic_dataset
from train import train_model

# Constants for Styling
BG_COLOR = "#121212"
CARD_BG = "#1E1E1E"
TEXT_COLOR = "#E0E0E0"
ACCENT_BLUE = "#3498DB"
ACCENT_GREEN = "#2ECC71"
ACCENT_RED = "#E74C3C"
FONT_FAMILY = "Segoe UI"

class UC_DFNet_DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UC-DFNet Steganalysis Desktop Portal")
        self.root.geometry("1200x850")
        self.root.configure(bg=BG_COLOR)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Modify TNotebook styling
        self.style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=CARD_BG, foreground=TEXT_COLOR, padding=[15, 5], font=(FONT_FAMILY, 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", ACCENT_BLUE)], foreground=[("selected", "white")])
        
        # Initialize variables
        self.model_path = "best_model.pth"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selected_img_path = None
        self.original_cv_img = None
        self.processed_stego_img = None
        
        # Main Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create Tabs
        self.create_steganalysis_tab()
        self.create_sandbox_tab()
        self.create_training_tab()
        
        # Load model if it exists
        self.load_model_weights()
        
    def load_model_weights(self):
        if os.path.exists(self.model_path):
            try:
                self.model = UCDFNet(num_classes=2)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.status_label.config(text=f"Model Loaded successfully (Device: {self.device})", foreground=ACCENT_GREEN)
            except Exception as e:
                self.status_label.config(text=f"Error loading model weights: {e}", foreground=ACCENT_RED)
        else:
            self.status_label.config(text="Warning: 'best_model.pth' not found. Please train the model in the Training Wizard.", foreground=ACCENT_RED)

    # ------------------ TAB 1: STEGANALYSIS ------------------
    def create_steganalysis_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🔍 Steganalysis Dashboard")
        
        # Grid layout: Left control & Right result
        tab.columnconfigure(0, weight=1, minsize=400)
        tab.columnconfigure(1, weight=1, minsize=400)
        tab.rowconfigure(0, weight=1)
        
        # Left Panel (Input / Image Preview)
        left_panel = tk.Frame(tab, bg=CARD_BG, bd=1, relief="flat")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        lbl_input = tk.Label(left_panel, text="1. Select Input Image", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 14, "bold"))
        lbl_input.pack(anchor="w", padx=20, pady=15)
        
        btn_select = tk.Button(left_panel, text="📁 Select Color Image", bg=ACCENT_BLUE, fg="white", activebackground="#2980B9", activeforeground="white", bd=0, padx=15, pady=8, font=(FONT_FAMILY, 10, "bold"), command=self.select_image)
        btn_select.pack(fill="x", padx=40, pady=5)
        
        self.img_preview_lbl = tk.Label(left_panel, text="No Image Selected\n(Preview will show here)", bg=BG_COLOR, fg=TEXT_COLOR, width=40, height=18, font=(FONT_FAMILY, 10))
        self.img_preview_lbl.pack(fill="both", expand=True, padx=40, pady=20)
        
        self.btn_analyze = tk.Button(left_panel, text="🔎 Run UC-DFNet Steganalysis", bg=ACCENT_GREEN, fg="white", activebackground="#27AE60", activeforeground="white", bd=0, padx=15, pady=10, font=(FONT_FAMILY, 11, "bold"), state="disabled", command=self.analyze_image)
        self.btn_analyze.pack(fill="x", padx=40, pady=15)
        
        # Right Panel (Inference & Grad-CAM outputs)
        right_panel = tk.Frame(tab, bg=CARD_BG, bd=1, relief="flat")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        lbl_result_hdr = tk.Label(right_panel, text="2. Analysis Results", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 14, "bold"))
        lbl_result_hdr.pack(anchor="w", padx=20, pady=15)
        
        # Result Notification Card
        self.res_card = tk.Frame(right_panel, bg=BG_COLOR, height=100, bd=1)
        self.res_card.pack(fill="x", padx=30, pady=10)
        self.res_card.pack_propagate(False)
        
        self.res_lbl = tk.Label(self.res_card, text="Awaiting Image Analysis...", bg=BG_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, 14, "bold"))
        self.res_lbl.pack(fill="both", expand=True)
        
        # Grad-CAM Frame
        gradcam_frame = tk.Frame(right_panel, bg=CARD_BG)
        gradcam_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        lbl_gradcam = tk.Label(gradcam_frame, text="Grad-CAM Spatial Explainability Heatmap:", bg=CARD_BG, fg=TEXT_COLOR, font=(FONT_FAMILY, 11, "bold"))
        lbl_gradcam.pack(anchor="w", pady=5)
        
        self.gradcam_preview = tk.Label(gradcam_frame, text="Explainability visual overlay will render here", bg=BG_COLOR, fg=TEXT_COLOR, width=40, height=14, font=(FONT_FAMILY, 10))
        self.gradcam_preview.pack(fill="both", expand=True)
        
        # Status Bar at Bottom
        self.status_label = tk.Label(self.root, text="System Ready", bg=BG_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, 9), anchor="w", padx=15, pady=5)
        self.status_label.pack(side="bottom", fill="x")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.selected_img_path = file_path
            
            # Display image in preview
            img = Image.open(file_path)
            img.thumbnail((350, 350))
            self.preview_img_tk = ImageTk.PhotoImage(img)
            self.img_preview_lbl.config(image=self.preview_img_tk, text="")
            
            self.status_label.config(text=f"Loaded image: {file_path}", foreground=TEXT_COLOR)
            self.btn_analyze.config(state="normal")
            
            # Clear previous results
            self.res_lbl.config(text="Ready to analyze.", bg=BG_COLOR, fg=TEXT_COLOR)
            self.res_card.config(bg=BG_COLOR)
            self.gradcam_preview.config(image="", text="Explainability visual overlay will render here")

    def analyze_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model weights not loaded. Please train or supply 'best_model.pth'.")
            return
            
        self.status_label.config(text="Analyzing image, please wait...")
        self.btn_analyze.config(state="disabled")
        
        # Run inference in a background thread to prevent UI freezing
        thread = threading.Thread(target=self._run_inference_thread)
        thread.start()
        
    def _run_inference_thread(self):
        try:
            # Preprocess image
            image_tensor, original_rgb = preprocess_image(self.selected_img_path)
            
            # Run inference
            pred_label, confidence, probs = run_inference(self.model, image_tensor, self.device)
            
            # Grad-CAM calculations
            grad_cam = GradCAM(self.model, self.model.stage3_fdb)
            pred_class = np.argmax(probs)
            cam_np, _, _ = grad_cam.generate_cam(image_tensor.to(self.device), target_class=pred_class)
            overlaid, _ = get_heatmap_overlay(original_rgb, cam_np)
            grad_cam.remove_hooks()
            
            # Update UI on main thread
            self.root.after(0, self._update_analysis_ui, pred_label, confidence, overlaid)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Inference Error", f"Failed running inference: {e}"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    def _update_analysis_ui(self, pred_label, confidence, overlaid_img):
        # Update prediction cards
        is_stego = "Stego" in pred_label
        bg_card = ACCENT_RED if is_stego else ACCENT_GREEN
        fg_card = "white"
        
        status_text = f"{pred_label.upper()}\nConfidence: {confidence * 100:.2f}%"
        self.res_lbl.config(text=status_text, bg=bg_card, fg=fg_card)
        self.res_card.config(bg=bg_card)
        
        # Display Grad-CAM overlay
        pil_overlaid = Image.fromarray(overlaid_img)
        pil_overlaid.thumbnail((350, 300))
        self.gradcam_img_tk = ImageTk.PhotoImage(pil_overlaid)
        self.gradcam_preview.config(image=self.gradcam_img_tk, text="")
        
        self.status_label.config(text="Analysis complete!", foreground=ACCENT_GREEN)
        self.btn_analyze.config(state="normal")


    # ------------------ TAB 2: STEGANOGRAPHY SANDBOX ------------------
    def create_sandbox_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🎨 Steganography Sandbox")
        
        tab.columnconfigure(0, weight=1, minsize=400)
        tab.columnconfigure(1, weight=1, minsize=400)
        tab.rowconfigure(0, weight=1)
        
        # Left sandbox panel: Embed
        embed_panel = tk.Frame(tab, bg=CARD_BG, bd=1)
        embed_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        lbl_embed = tk.Label(embed_panel, text="📤 Embed Secret Message", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 14, "bold"))
        lbl_embed.pack(anchor="w", padx=20, pady=15)
        
        btn_sel_cov = tk.Button(embed_panel, text="📁 Choose Cover Image (PNG)", bg=ACCENT_BLUE, fg="white", bd=0, padx=10, pady=5, font=(FONT_FAMILY, 9, "bold"), command=self.select_sandbox_cover)
        btn_sel_cov.pack(fill="x", padx=45, pady=5)
        
        self.cov_lbl = tk.Label(embed_panel, text="No cover image loaded", bg=CARD_BG, fg=TEXT_COLOR, font=(FONT_FAMILY, 9, "italic"))
        self.cov_lbl.pack(anchor="w", padx=45, pady=2)
        
        # Select Algorithm
        tk.Label(embed_panel, text="Algorithm:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(10, 2))
        self.embed_algo_var = tk.StringVar(value="LSB Replacement")
        algo_cb = ttk.Combobox(embed_panel, textvariable=self.embed_algo_var, values=["LSB Replacement", "LSB Matching", "Random Path LSB", "DCT Domain LSB"], state="readonly")
        algo_cb.pack(fill="x", padx=45, pady=2)
        
        # Select Channels
        tk.Label(embed_panel, text="Channels:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(10, 2))
        self.embed_chan_var = tk.StringVar(value="All Channels (RGB)")
        chan_cb = ttk.Combobox(embed_panel, textvariable=self.embed_chan_var, values=["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"], state="readonly")
        chan_cb.pack(fill="x", padx=45, pady=2)
        
        # Key / Parameter Input
        self.param_frame = tk.Frame(embed_panel, bg=CARD_BG)
        self.param_frame.pack(fill="x", padx=45, pady=10)
        
        tk.Label(self.param_frame, text="Secret Key / Q Value:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self.embed_param_val = tk.Entry(self.param_frame, bg=BG_COLOR, fg="white", insertbackground="white", bd=1, width=15)
        self.embed_param_val.pack(side="left", padx=10)
        self.embed_param_val.insert(0, "42")
        
        # Text Message Input
        tk.Label(embed_panel, text="Secret Text Message:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(10, 2))
        self.embed_msg_txt = tk.Entry(embed_panel, bg=BG_COLOR, fg="white", insertbackground="white", bd=1, font=(FONT_FAMILY, 10))
        self.embed_msg_txt.pack(fill="x", padx=45, pady=2)
        self.embed_msg_txt.insert(0, "Secret steganography payload!")
        
        btn_embed_exec = tk.Button(embed_panel, text="🔒 Embed and Export Stego Image", bg=ACCENT_GREEN, fg="white", bd=0, padx=10, pady=10, font=(FONT_FAMILY, 10, "bold"), command=self.run_embedding)
        btn_embed_exec.pack(fill="x", padx=45, pady=20)
        
        # Right sandbox panel: Extract
        extract_panel = tk.Frame(tab, bg=CARD_BG, bd=1)
        extract_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        lbl_extract = tk.Label(extract_panel, text="🔑 Extract Secret Message", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 14, "bold"))
        lbl_extract.pack(anchor="w", padx=20, pady=15)
        
        btn_sel_steg = tk.Button(extract_panel, text="📁 Choose Stego Image (PNG)", bg=ACCENT_BLUE, fg="white", bd=0, padx=10, pady=5, font=(FONT_FAMILY, 9, "bold"), command=self.select_sandbox_stego)
        btn_sel_steg.pack(fill="x", padx=45, pady=5)
        
        self.steg_lbl = tk.Label(extract_panel, text="No stego image loaded", bg=CARD_BG, fg=TEXT_COLOR, font=(FONT_FAMILY, 9, "italic"))
        self.steg_lbl.pack(anchor="w", padx=45, pady=2)
        
        # Select Algorithm
        tk.Label(extract_panel, text="Expected Algorithm:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(10, 2))
        self.extract_algo_var = tk.StringVar(value="LSB Replacement / Matching")
        algo_cb_ext = ttk.Combobox(extract_panel, textvariable=self.extract_algo_var, values=["LSB Replacement / Matching", "Random Path LSB", "DCT Domain LSB"], state="readonly")
        algo_cb_ext.pack(fill="x", padx=45, pady=2)
        
        # Select Channels
        tk.Label(extract_panel, text="Expected Channels:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(10, 2))
        self.extract_chan_var = tk.StringVar(value="All Channels (RGB)")
        chan_cb_ext = ttk.Combobox(extract_panel, textvariable=self.extract_chan_var, values=["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"], state="readonly")
        chan_cb_ext.pack(fill="x", padx=45, pady=2)
        
        # Key / Parameter Input
        self.param_frame_ext = tk.Frame(extract_panel, bg=CARD_BG)
        self.param_frame_ext.pack(fill="x", padx=45, pady=10)
        
        tk.Label(self.param_frame_ext, text="Secret Key / Q Value:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self.extract_param_val = tk.Entry(self.param_frame_ext, bg=BG_COLOR, fg="white", insertbackground="white", bd=1, width=15)
        self.extract_param_val.pack(side="left", padx=10)
        self.extract_param_val.insert(0, "42")
        
        btn_extract_exec = tk.Button(extract_panel, text="🔓 Run Extraction", bg=ACCENT_RED, fg="white", bd=0, padx=10, pady=10, font=(FONT_FAMILY, 10, "bold"), command=self.run_extraction)
        btn_extract_exec.pack(fill="x", padx=45, pady=20)
        
        # Extracted Text Display Box
        tk.Label(extract_panel, text="Extracted Secret Message:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=45, pady=(5, 2))
        self.extracted_res_txt = tk.Text(extract_panel, height=4, bg=BG_COLOR, fg=ACCENT_GREEN, insertbackground="white", bd=1, font=(FONT_FAMILY, 10, "bold"))
        self.extracted_res_txt.pack(fill="x", padx=45, pady=5)

    def select_sandbox_cover(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.cover_img_path = file_path
            self.cov_lbl.config(text=os.path.basename(file_path), font=(FONT_FAMILY, 9, "bold"), fg="white")
            self.original_cv_img = cv2.imread(file_path)

    def select_sandbox_stego(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
        if file_path:
            self.stego_img_path = file_path
            self.steg_lbl.config(text=os.path.basename(file_path), font=(FONT_FAMILY, 9, "bold"), fg="white")
            self.processed_stego_img = cv2.imread(file_path)

    def run_embedding(self):
        if self.original_cv_img is None:
            messagebox.showerror("Error", "Please load a cover image first.")
            return
            
        message = self.embed_msg_txt.get()
        if not message:
            messagebox.showerror("Error", "Please enter a message to embed.")
            return
            
        algo = self.embed_algo_var.get()
        channel_desc = self.embed_chan_var.get()
        
        channel_map = {
            "All Channels (RGB)": [0, 1, 2],
            "Red Channel Only": [0],
            "Green Channel Only": [1],
            "Blue Channel Only": [2]
        }
        ch_list = channel_map[channel_desc]
        
        try:
            param = float(self.embed_param_val.get())
        except ValueError:
            messagebox.showerror("Error", "Secret key or parameter must be a valid number.")
            return
            
        try:
            # We perform embedding (OpenCV image is BGR, we pass it directly)
            if algo == "LSB Replacement":
                stego = embed_lsb(self.original_cv_img, message, channels=ch_list)
            elif algo == "LSB Matching":
                stego = embed_lsb_matching(self.original_cv_img, message, channels=ch_list)
            elif algo == "Random Path LSB":
                stego = embed_random_path(self.original_cv_img, message, key=int(param), channels=ch_list)
            else: # DCT
                stego = embed_dct(self.original_cv_img, message, channels=ch_list, Q=float(param))
                
            # Ask where to save
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if save_path:
                cv2.imwrite(save_path, stego)
                messagebox.showinfo("Success", f"Stego image successfully saved to:\n{save_path}")
                self.status_label.config(text=f"Exported stego: {save_path}", foreground=ACCENT_GREEN)
        except Exception as e:
            messagebox.showerror("Embedding Error", f"Failed to embed message: {e}")

    def run_extraction(self):
        if self.processed_stego_img is None:
            messagebox.showerror("Error", "Please load a stego image first.")
            return
            
        algo = self.extract_algo_var.get()
        channel_desc = self.extract_chan_var.get()
        
        channel_map = {
            "All Channels (RGB)": [0, 1, 2],
            "Red Channel Only": [0],
            "Green Channel Only": [1],
            "Blue Channel Only": [2]
        }
        ch_list = channel_map[channel_desc]
        
        try:
            param = float(self.extract_param_val.get())
        except ValueError:
            messagebox.showerror("Error", "Secret key or parameter must be a valid number.")
            return
            
        try:
            self.extracted_res_txt.delete("1.0", tk.END)
            self.status_label.config(text="Running payload extraction...")
            
            if algo == "LSB Replacement / Matching":
                extracted = extract_lsb(self.processed_stego_img, channels=ch_list)
            elif algo == "Random Path LSB":
                extracted = extract_random_path(self.processed_stego_img, key=int(param), channels=ch_list)
            else: # DCT
                extracted = extract_dct(self.processed_stego_img, channels=ch_list, Q=float(param))
                
            if extracted:
                self.extracted_res_txt.insert(tk.END, extracted)
                self.status_label.config(text="Extraction complete!", foreground=ACCENT_GREEN)
            else:
                self.extracted_res_txt.insert(tk.END, "(No message extracted or null string returned)")
                self.status_label.config(text="Extraction finished - no payload found.", foreground=ACCENT_BLUE)
        except Exception as e:
            messagebox.showerror("Extraction Error", f"Failed to extract payload: {e}")


    # ------------------ TAB 3: MODEL TRAINING WIZARD ------------------
    def create_training_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🎓 Model Training Wizard")
        
        panel = tk.Frame(tab, bg=CARD_BG, bd=1)
        panel.pack(fill="both", expand=True, padx=20, pady=20)
        
        lbl_train = tk.Label(panel, text="🎓 UC-DFNet Local Network Training", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 14, "bold"))
        lbl_train.pack(anchor="w", padx=20, pady=15)
        
        tk.Label(panel, text="Train your local steganalysis model on custom textured datasets using your CPU/GPU.\nThe training wizard will synthesize cover/stego pairs automatically using a mixture of algorithms.", bg=CARD_BG, fg=TEXT_COLOR, justify="left", font=(FONT_FAMILY, 10)).pack(anchor="w", padx=30, pady=5)
        
        # Configure parameters
        config_frame = tk.Frame(panel, bg=CARD_BG)
        config_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(config_frame, text="Samples per class:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.train_samples_ent = tk.Entry(config_frame, bg=BG_COLOR, fg="white", bd=1, width=10)
        self.train_samples_ent.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        self.train_samples_ent.insert(0, "50")
        
        tk.Label(config_frame, text="Training Epochs:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.train_epochs_ent = tk.Entry(config_frame, bg=BG_COLOR, fg="white", bd=1, width=10)
        self.train_epochs_ent.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        self.train_epochs_ent.insert(0, "5")
        
        tk.Label(config_frame, text="Batch Size:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        self.train_batch_ent = tk.Entry(config_frame, bg=BG_COLOR, fg="white", bd=1, width=10)
        self.train_batch_ent.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        self.train_batch_ent.insert(0, "8")
        
        self.btn_run_train = tk.Button(panel, text="🚀 Generate Synthetic Dataset & Start Training", bg=ACCENT_BLUE, fg="white", bd=0, padx=15, pady=10, font=(FONT_FAMILY, 11, "bold"), command=self.start_training_process)
        self.btn_run_train.pack(fill="x", padx=30, pady=15)
        
        # Training Console Log
        tk.Label(panel, text="Training Progress Output Console:", bg=CARD_BG, fg="white", font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=30, pady=(10, 2))
        self.console_box = tk.Text(panel, height=12, bg=BG_COLOR, fg="#A2D2FF", bd=1, font=("Consolas", 9))
        self.console_box.pack(fill="both", expand=True, padx=30, pady=5)
        
    def log_to_console(self, text):
        self.console_box.insert(tk.END, text + "\n")
        self.console_box.see(tk.END)
        
    def start_training_process(self):
        self.btn_run_train.config(state="disabled")
        self.console_box.delete("1.0", tk.END)
        
        # Read parameters
        try:
            samples = int(self.train_samples_ent.get())
            epochs = int(self.train_epochs_ent.get())
            batch = int(self.train_batch_ent.get())
        except ValueError:
            messagebox.showerror("Error", "All training parameters must be valid integers.")
            self.btn_run_train.config(state="normal")
            return
            
        self.log_to_console("Initializing training wizard...")
        self.status_label.config(text="Training model, see console...")
        
        # Run training in background thread
        thread = threading.Thread(target=self._run_training_thread, args=(samples, epochs, batch))
        thread.start()
        
    def _run_training_thread(self, samples, epochs, batch):
        class StdoutRedirector:
            def __init__(self, app_instance):
                self.app = app_instance
            def write(self, s):
                if s.strip():
                    self.app.root.after(0, self.app.log_to_console, s.strip())
            def flush(self):
                pass
                
        # Redirect prints to console box
        old_stdout = os.sys.stdout
        os.sys.stdout = StdoutRedirector(self)
        
        try:
            # 1. Dataset Generation
            print("Generating synthetic cover/stego image texture pairs...")
            generate_synthetic_dataset("data", num_samples=samples)
            
            # 2. Setup training args class
            class TrainArgs:
                def __init__(self):
                    self.data_dir = "data"
                    self.num_samples = samples
                    self.epochs = epochs
                    self.batch_size = batch
                    self.lr = 0.001
                    self.patience = 5
                    self.model_path = "best_model.pth"
                    
            args = TrainArgs()
            
            # 3. Start training
            train_model(args)
            
            print("\nModel trained successfully!")
            self.root.after(0, self._finalize_training_success)
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training failed: {e}"))
            self.root.after(0, lambda: self.btn_run_train.config(state="normal"))
        finally:
            os.sys.stdout = old_stdout

    def _finalize_training_success(self):
        self.btn_run_train.config(state="normal")
        self.status_label.config(text="Model trained and saved locally!", foreground=ACCENT_GREEN)
        
        # Reload model weights into steganalysis tab
        self.load_model_weights()
        
        # Notify user
        messagebox.showinfo("Success", "UC-DFNet model trained successfully on local synthetic dataset! Model weights saved as 'best_model.pth'.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UC_DFNet_DesktopApp(root)
    root.mainloop()
