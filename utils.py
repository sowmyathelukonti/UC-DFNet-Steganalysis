import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def string_to_bits(s: str) -> list:
    """Convert a string to a list of bits (0 or 1)."""
    bits = []
    for char in s:
        # Convert character to 8-bit binary representation
        bin_val = bin(ord(char))[2:].zfill(8)
        bits.extend([int(b) for b in bin_val])
    # Add a null terminator (8 bits of 0) to mark the end of message
    bits.extend([0] * 8)
    return bits

def bits_to_string(bits: list) -> str:
    """Convert a list of bits back to a string, stopping at null terminator."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char_val = int("".join(map(str, byte)), 2)
        if char_val == 0:
            break  # Null terminator reached
        chars.append(chr(char_val))
    return "".join(chars)

def embed_lsb(image: np.ndarray, message: str) -> np.ndarray:
    """
    Embed a text message into the Least Significant Bits of a color image.
    Supports 3-channel color images (BGR/RGB).
    """
    stego_image = image.copy()
    bits = string_to_bits(message)
    total_bits = len(bits)
    
    h, w, c = stego_image.shape
    total_pixels = h * w * c
    
    if total_bits > total_pixels:
        raise ValueError(f"Message too long! Requires {total_bits} bits, but image only has capacity for {total_pixels} bits.")
    
    bit_idx = 0
    # Flatten the image loop for embedding
    for y in range(h):
        for x in range(w):
            for channel in range(c):
                if bit_idx >= total_bits:
                    return stego_image
                # Clear the LSB and set it to the message bit
                val = int(stego_image[y, x, channel])
                val = (val & ~1) | bits[bit_idx]
                stego_image[y, x, channel] = np.clip(val, 0, 255).astype(np.uint8)
                bit_idx += 1
                
    return stego_image

def extract_lsb(image: np.ndarray) -> str:
    """
    Extract a hidden text message from the LSBs of a color image.
    Stops extracting when the null terminator (8 consecutive zero bits) is found.
    """
    h, w, c = image.shape
    bits = []
    
    # We will check for the null terminator byte on the fly
    current_byte_bits = []
    
    for y in range(h):
        for x in range(w):
            for channel in range(c):
                # Get the LSB
                val = int(image[y, x, channel])
                bit = val & 1
                bits.append(bit)
                current_byte_bits.append(bit)
                
                # Check for null terminator after every full byte (8 bits)
                if len(current_byte_bits) == 8:
                    byte_val = int("".join(map(str, current_byte_bits)), 2)
                    if byte_val == 0:
                        # Null terminator found, stop extracting
                        return bits_to_string(bits)
                    current_byte_bits = []
                    
    return bits_to_string(bits)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir="."):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss', color='#4A90E2')
    plt.plot(epochs, val_losses, 's-', label='Val Loss', color='#E24A84')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'o-', label='Train Acc', color='#4AE290')
    plt.plot(epochs, val_accs, 's-', label='Val Acc', color='#E29B4A')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """Plot and save a stylized confusion matrix."""
    import itertools
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=300)
    plt.close()
