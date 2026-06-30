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

# Modern Color Palette (Deep Slate & Vibrant Accents)
BG_COLOR = "#0B0F19"         # Dark Space Blue
CARD_BG = "#131C2E"          # Dark Card Blue
CARD_BORDER = "#1E293B"      # Border Slate
TEXT_MAIN = "#F1F5F9"        # Main White
TEXT_MUTED = "#94A3B8"       # Muted Grey
ACCENT_BLUE = "#0284C7"      # Cyan Blue
ACCENT_BLUE_HOVER = "#0369A1"
ACCENT_PURPLE = "#8B5CF6"    # Vibrant Violet
ACCENT_PURPLE_HOVER = "#7C3AED"
ACCENT_GREEN = "#10B981"     # Emerald Green
ACCENT_GREEN_HOVER = "#059669"
ACCENT_RED = "#EF4444"       # Rose Red
ACCENT_RED_HOVER = "#DC2626"
CONSOLE_BG = "#070A13"       # Terminal Black

FONT_FAMILY = "Segoe UI"

class ModernButton(tk.Button):
    """Custom styled button with hover animations."""
    def __init__(self, master, hover_bg, normal_bg, **kwargs):
        super().__init__(master, **kwargs)
        self.hover_bg = hover_bg
        self.normal_bg = normal_bg
        self.config(
            bg=self.normal_bg,
            activebackground=self.hover_bg,
            bd=0,
            relief="flat",
            cursor="hand2"
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.hover_bg)

    def on_leave(self, e):
        self.config(bg=self.normal_bg)


class UC_DFNet_DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UC-DFNet Steganalysis Desktop Portal")
        self.root.geometry("1280x880")
        self.root.configure(bg=BG_COLOR)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure Notebook Tab Layouts
        self.style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=CARD_BG, foreground=TEXT_MUTED, 
                             padding=[20, 8], font=(FONT_FAMILY, 11, "bold"), borderwidth=0)
        self.style.map("TNotebook.Tab", 
                       background=[("selected", ACCENT_PURPLE)], 
                       foreground=[("selected", "white")])
        
        # Configure Combobox Styles
        self.style.configure("TCombobox", fieldbackground=BG_COLOR, background=CARD_BG, 
                             foreground=TEXT_MAIN, arrowcolor=TEXT_MUTED)
        self.style.map("TCombobox", fieldbackground=[("readonly", BG_COLOR)], foreground=[("readonly", TEXT_MAIN)])
        
        # Initialize variables
        self.model_path = "best_model.pth"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selected_img_path = None
        self.original_cv_img = None
        self.processed_stego_img = None
        
        # Header Branding Banner
        self.create_header_banner()
        
        # Main Notebook Container
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # Create Tabs
        self.create_steganalysis_tab()
        self.create_sandbox_tab()
        self.create_training_tab()
        
        # Status Bar at the Bottom
        self.create_status_bar()
        
        # Load model weights
        self.load_model_weights()

    def create_header_banner(self):
        header_frame = tk.Frame(self.root, bg=BG_COLOR, height=75)
        header_frame.pack(fill="x", padx=20, pady=10)
        header_frame.pack_propagate(False)
        
        # Logo Icon placeholder
        logo_label = tk.Label(header_frame, text="🛡️", bg=BG_COLOR, fg=ACCENT_PURPLE, font=(FONT_FAMILY, 24))
        logo_label.pack(side="left", padx=(0, 10))
        
        title_container = tk.Frame(header_frame, bg=BG_COLOR)
        title_container.pack(side="left")
        
        title_lbl = tk.Label(title_container, text="UC-DFNet Steganalysis Dashboard", bg=BG_COLOR, fg=TEXT_MAIN, 
                             font=(FONT_FAMILY, 16, "bold"))
        title_lbl.pack(anchor="w")
        
        sub_lbl = tk.Label(title_container, text="Universal Color Dual-Path Fractal Network for Color Image Steganalysis & Visual Interpretability", 
                           bg=BG_COLOR, fg=TEXT_MUTED, font=(FONT_FAMILY, 9))
        sub_lbl.pack(anchor="w")
        
        # Small device badge
        self.device_badge = tk.Label(header_frame, text=f"DEVICE: {str(self.device).upper()}", bg=CARD_BG, fg=ACCENT_PURPLE,
                                     padx=10, pady=5, font=(FONT_FAMILY, 8, "bold"), bd=1, relief="flat")
        self.device_badge.pack(side="right", padx=10)

    def create_status_bar(self):
        status_frame = tk.Frame(self.root, bg=BG_COLOR, height=30)
        status_frame.pack(side="bottom", fill="x", padx=20)
        
        self.status_label = tk.Label(status_frame, text="System Ready", bg=BG_COLOR, fg=TEXT_MUTED, 
                                     font=(FONT_FAMILY, 9), anchor="w")
        self.status_label.pack(fill="both", expand=True)

    def load_model_weights(self):
        if os.path.exists(self.model_path):
            try:
                self.model = UCDFNet(num_classes=2)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.status_label.config(text=f"UC-DFNet model weights loaded successfully (Device: {self.device})", foreground=ACCENT_GREEN)
            except Exception as e:
                self.status_label.config(text=f"Error loading model weights: {e}", foreground=ACCENT_RED)
        else:
            self.status_label.config(text="Warning: 'best_model.pth' not found. Please train the model in the Training Wizard tab.", foreground=ACCENT_RED)

    # ------------------ TAB 1: STEGANALYSIS ------------------
    def create_steganalysis_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🔍 Steganalysis Dashboard")
        
        # Layout splitting: Left and Right Panel
        tab.columnconfigure(0, weight=1, minsize=500)
        tab.columnconfigure(1, weight=1, minsize=500)
        tab.rowconfigure(0, weight=1)
        
        # Left Panel (Input selection)
        left_panel = tk.Frame(tab, bg=CARD_BG, bd=1, highlightbackground=CARD_BORDER, highlightthickness=1)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        lbl_hdr = tk.Label(left_panel, text="1. Select Input Image", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold"))
        lbl_hdr.pack(anchor="w", padx=25, pady=(20, 10))
        
        lbl_desc = tk.Label(left_panel, text="Choose a color image file to scan for hidden stego payloads.", bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_FAMILY, 9))
        lbl_desc.pack(anchor="w", padx=25, pady=(0, 15))
        
        # Styled Image drop box canvas
        self.upload_box = tk.Canvas(left_panel, bg=BG_COLOR, bd=0, highlightthickness=1, highlightbackground=CARD_BORDER)
        self.upload_box.pack(fill="both", expand=True, padx=40, pady=10)
        
        self.upload_inner_lbl = tk.Label(self.upload_box, text="No Image Selected\n\nPreview will show here after loading", bg=BG_COLOR, fg=TEXT_MUTED, font=(FONT_FAMILY, 10))
        self.upload_inner_lbl.place(relx=0.5, rely=0.5, anchor="center")
        
        # Action Buttons frame
        btn_frame = tk.Frame(left_panel, bg=CARD_BG)
        btn_frame.pack(fill="x", padx=40, pady=25)
        
        self.btn_select = ModernButton(btn_frame, ACCENT_BLUE_HOVER, ACCENT_BLUE, text="📁 Choose File", fg="white", 
                                       font=(FONT_FAMILY, 10, "bold"), height=2, command=self.select_image)
        self.btn_select.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_analyze = ModernButton(btn_frame, ACCENT_GREEN_HOVER, ACCENT_GREEN, text="🔎 Scan Image", fg="white", 
                                        font=(FONT_FAMILY, 10, "bold"), height=2, state="disabled", command=self.analyze_image)
        self.btn_analyze.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        # Right Panel (Results & Grad-CAM)
        right_panel = tk.Frame(tab, bg=CARD_BG, bd=1, highlightbackground=CARD_BORDER, highlightthickness=1)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
        lbl_hdr_res = tk.Label(right_panel, text="2. Detection Results", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold"))
        lbl_hdr_res.pack(anchor="w", padx=25, pady=(20, 10))
        
        # Prediction Output Card
        self.res_card = tk.Frame(right_panel, bg=BG_COLOR, height=110, highlightthickness=1, highlightbackground=CARD_BORDER)
        self.res_card.pack(fill="x", padx=40, pady=10)
        self.res_card.pack_propagate(False)
        
        self.res_lbl = tk.Label(self.res_card, text="Awaiting Image Scanning...", bg=BG_COLOR, fg=TEXT_MUTED, font=(FONT_FAMILY, 13, "bold"))
        self.res_lbl.pack(fill="both", expand=True)
        
        # Grad-CAM explaining frame
        grad_frame = tk.Frame(right_panel, bg=CARD_BG)
        grad_frame.pack(fill="both", expand=True, padx=40, pady=(15, 25))
        
        lbl_explain = tk.Label(grad_frame, text="Grad-CAM Stego Localization Heatmap:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold"))
        lbl_explain.pack(anchor="w", pady=(0, 10))
        
        self.gradcam_canvas = tk.Canvas(grad_frame, bg=BG_COLOR, bd=0, highlightthickness=1, highlightbackground=CARD_BORDER)
        self.gradcam_canvas.pack(fill="both", expand=True)
        
        self.gradcam_inner_lbl = tk.Label(self.gradcam_canvas, text="Grad-CAM analysis overlay will render here\nafter running steganalysis", bg=BG_COLOR, fg=TEXT_MUTED, font=(FONT_FAMILY, 9))
        self.gradcam_inner_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.selected_img_path = file_path
            
            # Open and scale image for preview box
            img = Image.open(file_path)
            self.upload_box.update()
            canvas_w = self.upload_box.winfo_width()
            canvas_h = self.upload_box.winfo_height()
            if canvas_w < 50: canvas_w = 400
            if canvas_h < 50: canvas_h = 350
            
            img.thumbnail((canvas_w, canvas_h))
            self.preview_img_tk = ImageTk.PhotoImage(img)
            self.upload_inner_lbl.config(image=self.preview_img_tk, text="")
            
            # Reset UI States
            self.res_lbl.config(text="IMAGE LOADED\nClick 'Scan Image' to start", bg=BG_COLOR, fg=TEXT_MAIN)
            self.res_card.config(bg=BG_COLOR, highlightbackground=CARD_BORDER)
            self.gradcam_inner_lbl.config(image="", text="Grad-CAM analysis overlay will render here\nafter running steganalysis")
            self.btn_analyze.config(state="normal")
            self.status_label.config(text=f"Loaded image: {file_path}", foreground=TEXT_MUTED)

    def analyze_image(self):
        if not self.model:
            messagebox.showerror("Error", "Model weights not loaded. Please train a network in the Training Wizard first.")
            return
            
        self.status_label.config(text="Processing network steganalysis, extracting features...")
        self.btn_analyze.config(state="disabled")
        
        # Run inference in a background thread to keep UI alive
        thread = threading.Thread(target=self._run_inference_thread)
        thread.start()
        
    def _run_inference_thread(self):
        try:
            image_tensor, original_rgb = preprocess_image(self.selected_img_path)
            
            # Run inference
            pred_label, confidence, probs = run_inference(self.model, image_tensor, self.device)
            
            # Run Grad-CAM
            grad_cam = GradCAM(self.model, self.model.stage3_fdb)
            pred_class = np.argmax(probs)
            cam_np, _, _ = grad_cam.generate_cam(image_tensor.to(self.device), target_class=pred_class)
            overlaid, _ = get_heatmap_overlay(original_rgb, cam_np)
            grad_cam.remove_hooks()
            
            # Update GUI on main thread
            self.root.after(0, self._update_analysis_ui, pred_label, confidence, overlaid)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed steganalysis run: {e}"))
            self.root.after(0, lambda: self.btn_analyze.config(state="normal"))

    def _update_analysis_ui(self, pred_label, confidence, overlaid_img):
        # Determine stego risk styling
        is_stego = "Stego" in pred_label
        bg = ACCENT_RED if is_stego else ACCENT_GREEN
        icon = "🚨" if is_stego else "🛡️"
        
        status_txt = f"{icon} {pred_label.upper()}\nRisk Probability: {confidence * 100:.2f}%"
        self.res_lbl.config(text=status_txt, bg=bg, fg="white")
        self.res_card.config(bg=bg, highlightbackground=bg)
        
        # Render Grad-CAM Heatmap
        pil_overlaid = Image.fromarray(overlaid_img)
        self.gradcam_canvas.update()
        canvas_w = self.gradcam_canvas.winfo_width()
        canvas_h = self.gradcam_canvas.winfo_height()
        if canvas_w < 50: canvas_w = 400
        if canvas_h < 50: canvas_h = 300
        
        pil_overlaid.thumbnail((canvas_w, canvas_h))
        self.gradcam_img_tk = ImageTk.PhotoImage(pil_overlaid)
        self.gradcam_inner_lbl.config(image=self.gradcam_img_tk, text="")
        
        self.status_label.config(text="Steganalysis complete successfully!", foreground=ACCENT_GREEN)
        self.btn_analyze.config(state="normal")


    # ------------------ TAB 2: STEGANOGRAPHY SANDBOX ------------------
    def create_sandbox_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🎨 Steganography Sandbox")
        
        tab.columnconfigure(0, weight=1, minsize=500)
        tab.columnconfigure(1, weight=1, minsize=500)
        tab.rowconfigure(0, weight=1)
        
        # Left Panel (Embed Payload)
        embed_panel = tk.Frame(tab, bg=CARD_BG, bd=1, highlightbackground=CARD_BORDER, highlightthickness=1)
        embed_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        lbl_embed = tk.Label(embed_panel, text="🔒 Embed Message", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold"))
        lbl_embed.pack(anchor="w", padx=25, pady=(20, 10))
        
        btn_sel_cov = ModernButton(embed_panel, ACCENT_BLUE_HOVER, ACCENT_BLUE, text="📁 Select Cover Image (PNG/JPG)", 
                                   fg="white", font=(FONT_FAMILY, 9, "bold"), height=2, command=self.select_sandbox_cover)
        btn_sel_cov.pack(fill="x", padx=40, pady=5)
        
        self.cov_lbl = tk.Label(embed_panel, text="No cover image loaded", bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_FAMILY, 9, "italic"))
        self.cov_lbl.pack(anchor="w", padx=40, pady=2)
        
        # Option Form Frame
        form_frame = tk.Frame(embed_panel, bg=CARD_BG)
        form_frame.pack(fill="both", expand=True, padx=40, pady=10)
        
        # Algorithm select
        tk.Label(form_frame, text="Steganography Method:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.embed_algo_var = tk.StringVar(value="LSB Replacement")
        algo_cb = ttk.Combobox(form_frame, textvariable=self.embed_algo_var, 
                              values=["LSB Replacement", "LSB Matching", "Random Path LSB", "DCT Domain LSB"], state="readonly")
        algo_cb.pack(fill="x", pady=2)
        
        # Channel Select
        tk.Label(form_frame, text="Active Color Channels:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.embed_chan_var = tk.StringVar(value="All Channels (RGB)")
        chan_cb = ttk.Combobox(form_frame, textvariable=self.embed_chan_var, 
                              values=["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"], state="readonly")
        chan_cb.pack(fill="x", pady=2)
        
        # Parameters Frame (Key or Quantization)
        param_f = tk.Frame(form_frame, bg=CARD_BG)
        param_f.pack(fill="x", pady=10)
        tk.Label(param_f, text="Key / Quantization (Q):", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self.embed_param_val = tk.Entry(param_f, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, width=15)
        self.embed_param_val.pack(side="left", padx=10)
        self.embed_param_val.insert(0, "42")
        
        # Message input
        tk.Label(form_frame, text="Message to Hide:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.embed_msg_txt = tk.Entry(form_frame, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, font=(FONT_FAMILY, 10))
        self.embed_msg_txt.pack(fill="x", pady=2)
        self.embed_msg_txt.insert(0, "Universal stego message! 🤫")
        
        btn_embed_exec = ModernButton(embed_panel, ACCENT_GREEN_HOVER, ACCENT_GREEN, text="🔒 Embed & Save Image", 
                                       fg="white", font=(FONT_FAMILY, 10, "bold"), height=2, command=self.run_embedding)
        btn_embed_exec.pack(fill="x", padx=40, pady=25)
        
        # Right Panel (Extract Payload)
        extract_panel = tk.Frame(tab, bg=CARD_BG, bd=1, highlightbackground=CARD_BORDER, highlightthickness=1)
        extract_panel.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
        lbl_extract = tk.Label(extract_panel, text="🔑 Extract Message", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 13, "bold"))
        lbl_extract.pack(anchor="w", padx=25, pady=(20, 10))
        
        btn_sel_steg = ModernButton(extract_panel, ACCENT_BLUE_HOVER, ACCENT_BLUE, text="📁 Select Stego Image (PNG)", 
                                   fg="white", font=(FONT_FAMILY, 9, "bold"), height=2, command=self.select_sandbox_stego)
        btn_sel_steg.pack(fill="x", padx=40, pady=5)
        
        self.steg_lbl = tk.Label(extract_panel, text="No stego image loaded", bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_FAMILY, 9, "italic"))
        self.steg_lbl.pack(anchor="w", padx=40, pady=2)
        
        # Form Extract Frame
        form_frame_ext = tk.Frame(extract_panel, bg=CARD_BG)
        form_frame_ext.pack(fill="both", expand=True, padx=40, pady=10)
        
        tk.Label(form_frame_ext, text="Expected Stego Method:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.extract_algo_var = tk.StringVar(value="LSB Replacement / Matching")
        algo_cb_ext = ttk.Combobox(form_frame_ext, textvariable=self.extract_algo_var, 
                                  values=["LSB Replacement / Matching", "Random Path LSB", "DCT Domain LSB"], state="readonly")
        algo_cb_ext.pack(fill="x", pady=2)
        
        tk.Label(form_frame_ext, text="Expected Channels:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.extract_chan_var = tk.StringVar(value="All Channels (RGB)")
        chan_cb_ext = ttk.Combobox(form_frame_ext, textvariable=self.extract_chan_var, 
                                  values=["All Channels (RGB)", "Red Channel Only", "Green Channel Only", "Blue Channel Only"], state="readonly")
        chan_cb_ext.pack(fill="x", pady=2)
        
        param_f_ext = tk.Frame(form_frame_ext, bg=CARD_BG)
        param_f_ext.pack(fill="x", pady=10)
        tk.Label(param_f_ext, text="Secret Key / Q Value:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        self.extract_param_val = tk.Entry(param_f_ext, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, width=15)
        self.extract_param_val.pack(side="left", padx=10)
        self.extract_param_val.insert(0, "42")
        
        btn_extract_exec = ModernButton(extract_panel, ACCENT_RED_HOVER, ACCENT_RED, text="🔓 Extract Message", 
                                       fg="white", font=(FONT_FAMILY, 10, "bold"), height=2, command=self.run_extraction)
        btn_extract_exec.pack(fill="x", padx=40, pady=15)
        
        # Result output box
        tk.Label(extract_panel, text="Extracted Secret Message:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=40, pady=(5, 2))
        
        self.extracted_res_txt = tk.Text(extract_panel, height=3, bg=CONSOLE_BG, fg=ACCENT_GREEN, insertbackground="white", 
                                         bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, font=("Consolas", 10, "bold"))
        self.extracted_res_txt.pack(fill="x", padx=40, pady=(0, 25))

    def select_sandbox_cover(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.cover_img_path = file_path
            self.cov_lbl.config(text=os.path.basename(file_path), font=(FONT_FAMILY, 9, "bold"), fg=TEXT_MAIN)
            self.original_cv_img = cv2.imread(file_path)

    def select_sandbox_stego(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
        if file_path:
            self.stego_img_path = file_path
            self.steg_lbl.config(text=os.path.basename(file_path), font=(FONT_FAMILY, 9, "bold"), fg=TEXT_MAIN)
            self.processed_stego_img = cv2.imread(file_path)

    def run_embedding(self):
        if self.original_cv_img is None:
            messagebox.showerror("Error", "Please select a cover image first.")
            return
            
        message = self.embed_msg_txt.get()
        if not message:
            messagebox.showerror("Error", "Please enter a message to hide.")
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
            if algo == "LSB Replacement":
                stego = embed_lsb(self.original_cv_img, message, channels=ch_list)
            elif algo == "LSB Matching":
                stego = embed_lsb_matching(self.original_cv_img, message, channels=ch_list)
            elif algo == "Random Path LSB":
                stego = embed_random_path(self.original_cv_img, message, key=int(param), channels=ch_list)
            else: # DCT
                stego = embed_dct(self.original_cv_img, message, channels=ch_list, Q=float(param))
                
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if save_path:
                cv2.imwrite(save_path, stego)
                messagebox.showinfo("Success", f"Stego image successfully saved!\nLocation: {save_path}")
                self.status_label.config(text=f"Exported stego: {save_path}", foreground=ACCENT_GREEN)
        except Exception as e:
            messagebox.showerror("Embedding Error", f"Failed embedding: {e}")

    def run_extraction(self):
        if self.processed_stego_img is None:
            messagebox.showerror("Error", "Please select a stego image first.")
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
            self.status_label.config(text="Running payload decoding...")
            
            if algo == "LSB Replacement / Matching":
                extracted = extract_lsb(self.processed_stego_img, channels=ch_list)
            elif algo == "Random Path LSB":
                extracted = extract_random_path(self.processed_stego_img, key=int(param), channels=ch_list)
            else: # DCT
                extracted = extract_dct(self.processed_stego_img, channels=ch_list, Q=float(param))
                
            if extracted:
                self.extracted_res_txt.insert(tk.END, extracted)
                self.status_label.config(text="Payload extracted successfully!", foreground=ACCENT_GREEN)
            else:
                self.extracted_res_txt.insert(tk.END, "(No message extracted or null string returned)")
                self.status_label.config(text="Finished: No message detected.", foreground=ACCENT_BLUE)
        except Exception as e:
            messagebox.showerror("Extraction Error", f"Extraction failed: {e}")


    # ------------------ TAB 3: MODEL TRAINING WIZARD ------------------
    def create_training_tab(self):
        tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(tab, text="🎓 Training Wizard")
        
        panel = tk.Frame(tab, bg=CARD_BG, bd=1, highlightbackground=CARD_BORDER, highlightthickness=1)
        panel.pack(fill="both", expand=True, padx=20, pady=20)
        
        lbl_train = tk.Label(panel, text="🎓 UC-DFNet Local Network Training Wizard", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 14, "bold"))
        lbl_train.pack(anchor="w", padx=25, pady=(20, 5))
        
        tk.Label(panel, text="Train your local steganalysis model on custom textured datasets using your CPU/GPU.\nThe training wizard will synthesize cover/stego pairs automatically using a mixture of algorithms.", bg=CARD_BG, fg=TEXT_MUTED, justify="left", font=(FONT_FAMILY, 10)).pack(anchor="w", padx=25, pady=5)
        
        # Parameters Input Area
        config_frame = tk.Frame(panel, bg=CARD_BG)
        config_frame.pack(fill="x", padx=25, pady=15)
        
        # Grid config
        config_frame.columnconfigure(0, weight=0)
        config_frame.columnconfigure(1, weight=1)
        
        tk.Label(config_frame, text="Samples per class:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.train_samples_ent = tk.Entry(config_frame, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, width=15)
        self.train_samples_ent.grid(row=0, column=1, sticky="w", padx=15, pady=5)
        self.train_samples_ent.insert(0, "50")
        
        tk.Label(config_frame, text="Training Epochs:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.train_epochs_ent = tk.Entry(config_frame, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, width=15)
        self.train_epochs_ent.grid(row=1, column=1, sticky="w", padx=15, pady=5)
        self.train_epochs_ent.insert(0, "5")
        
        tk.Label(config_frame, text="Batch Size:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        self.train_batch_ent = tk.Entry(config_frame, bg=BG_COLOR, fg=TEXT_MAIN, insertbackground="white", bd=0, highlightthickness=1, highlightbackground=CARD_BORDER, width=15)
        self.train_batch_ent.grid(row=2, column=1, sticky="w", padx=15, pady=5)
        self.train_batch_ent.insert(0, "8")
        
        self.btn_run_train = ModernButton(panel, ACCENT_PURPLE_HOVER, ACCENT_PURPLE, text="🚀 Start Training Network", 
                                         fg="white", font=(FONT_FAMILY, 11, "bold"), height=2, command=self.start_training_process)
        self.btn_run_train.pack(fill="x", padx=25, pady=10)
        
        # Training Console Terminal look
        tk.Label(panel, text="Training Console Output:", bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=25, pady=(15, 2))
        
        # Scrollbar for console
        console_frame = tk.Frame(panel, bg=CONSOLE_BG, bd=1, highlightthickness=1, highlightbackground=CARD_BORDER)
        console_frame.pack(fill="both", expand=True, padx=25, pady=(0, 25))
        
        self.console_box = tk.Text(console_frame, bg=CONSOLE_BG, fg="#38BDF8", bd=0, font=("Consolas", 9), wrap="word")
        self.console_box.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(console_frame, command=self.console_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.console_box.config(yscrollcommand=scrollbar.set)
        
    def log_to_console(self, text):
        self.console_box.insert(tk.END, text + "\n")
        self.console_box.see(tk.END)
        
    def start_training_process(self):
        self.btn_run_train.config(state="disabled")
        self.console_box.delete("1.0", tk.END)
        
        try:
            samples = int(self.train_samples_ent.get())
            epochs = int(self.train_epochs_ent.get())
            batch = int(self.train_batch_ent.get())
        except ValueError:
            messagebox.showerror("Error", "All parameters must be valid integers.")
            self.btn_run_train.config(state="normal")
            return
            
        self.log_to_console("Initializing training pipeline...")
        self.status_label.config(text="Model training in progress, check output console...")
        
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
            print("Generating synthetic cover/stego image texture dataset...")
            generate_synthetic_dataset("data", num_samples=samples)
            
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
            train_model(args)
            
            print("\nNetwork Training Completed Successfully!")
            self.root.after(0, self._finalize_training_success)
        except Exception as e:
            print(f"\nTraining interrupted with error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training aborted: {e}"))
            self.root.after(0, lambda: self.btn_run_train.config(state="normal"))
        finally:
            os.sys.stdout = old_stdout

    def _finalize_training_success(self):
        self.btn_run_train.config(state="normal")
        self.status_label.config(text="Training complete. Model weights saved!", foreground=ACCENT_GREEN)
        
        # Reload model weights into steganalysis tab
        self.load_model_weights()
        
        messagebox.showinfo("Success", "UC-DFNet model trained successfully!\nWeights saved to 'best_model.pth'.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UC_DFNet_DesktopApp(root)
    root.mainloop()
