import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def embed_lsb(image: np.ndarray, message: str, channels=[0, 1, 2]) -> np.ndarray:
    """
    Embed a text message into the Least Significant Bits of a color image
    for selected channels.
    """
    stego_image = image.copy()
    bits = string_to_bits(message)
    total_bits = len(bits)
    
    h, w, _ = stego_image.shape
    total_slots = h * w * len(channels)
    
    if total_bits > total_slots:
        raise ValueError(f"Message too long! Requires {total_bits} bits, but selection only has capacity for {total_slots} bits.")
    
    bit_idx = 0
    for y in range(h):
        for x in range(w):
            for c in channels:
                if bit_idx >= total_bits:
                    return stego_image
                val = int(stego_image[y, x, c])
                val = (val & ~1) | bits[bit_idx]
                stego_image[y, x, c] = np.clip(val, 0, 255).astype(np.uint8)
                bit_idx += 1
                
    return stego_image

def extract_lsb(image: np.ndarray, channels=[0, 1, 2]) -> str:
    """
    Extract a hidden text message from the LSBs of a color image
    from selected channels.
    """
    h, w, _ = image.shape
    bits = []
    current_byte_bits = []
    
    for y in range(h):
        for x in range(w):
            for c in channels:
                val = int(image[y, x, c])
                bit = val & 1
                bits.append(bit)
                current_byte_bits.append(bit)
                
                if len(current_byte_bits) == 8:
                    byte_val = int("".join(map(str, current_byte_bits)), 2)
                    if byte_val == 0:
                        return bits_to_string(bits)
                    current_byte_bits = []
                    
    return bits_to_string(bits)

def embed_lsb_matching(image: np.ndarray, message: str, channels=[0, 1, 2]) -> np.ndarray:
    """
    Embed a text message using LSB Matching (LSB ±1).
    Adds or subtracts 1 randomly when the LSB doesn't match the message bit.
    """
    import random
    stego_image = image.copy()
    bits = string_to_bits(message)
    total_bits = len(bits)
    
    h, w, _ = stego_image.shape
    total_slots = h * w * len(channels)
    
    if total_bits > total_slots:
        raise ValueError(f"Message too long! Requires {total_bits} bits, but selection only has capacity for {total_slots} bits.")
        
    random.seed(42)  # Set seed locally for reproducibility
    
    bit_idx = 0
    for y in range(h):
        for x in range(w):
            for c in channels:
                if bit_idx >= total_bits:
                    return stego_image
                val = int(stego_image[y, x, c])
                target_bit = bits[bit_idx]
                
                if (val & 1) != target_bit:
                    change = random.choice([-1, 1])
                    if val == 0:
                        change = 1
                    elif val == 255:
                        change = -1
                    val += change
                    
                stego_image[y, x, c] = np.clip(val, 0, 255).astype(np.uint8)
                bit_idx += 1
                
    return stego_image

def lcg_shuffle(lst, seed):
    """
    Linear Congruential Generator (LCG) shuffle.
    Guarantees cross-platform, bit-perfect identical shuffling between Python and JS.
    """
    state = seed
    shuffled = lst.copy()
    n = len(shuffled)
    for i in range(n - 1, 0, -1):
        state = (1664525 * state + 1013904223) % 4294967296
        j = state % (i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled

def embed_random_path(image: np.ndarray, message: str, key: int = 42, channels=[0, 1, 2]) -> np.ndarray:
    """
    Embed a message in a pseudo-random path of pixels determined by a secret key.
    """
    stego_image = image.copy()
    bits = string_to_bits(message)
    total_bits = len(bits)
    
    h, w, _ = stego_image.shape
    
    # Generate coordinates for selected channels
    coords = []
    for y in range(h):
        for x in range(w):
            for c in channels:
                coords.append((y, x, c))
                
    if total_bits > len(coords):
        raise ValueError(f"Message too long! Requires {total_bits} bits, but selection only has capacity for {len(coords)} bits.")
        
    coords = lcg_shuffle(coords, key)
    
    for bit_idx, (y, x, c) in enumerate(coords):
        if bit_idx >= total_bits:
            break
        val = int(stego_image[y, x, c])
        val = (val & ~1) | bits[bit_idx]
        stego_image[y, x, c] = np.clip(val, 0, 255).astype(np.uint8)
        
    return stego_image

def extract_random_path(image: np.ndarray, key: int = 42, channels=[0, 1, 2]) -> str:
    """
    Extract a message from a pseudo-random path of pixels determined by a secret key.
    """
    h, w, _ = image.shape
    
    coords = []
    for y in range(h):
        for x in range(w):
            for c in channels:
                coords.append((y, x, c))
                
    coords = lcg_shuffle(coords, key)
    
    bits = []
    current_byte_bits = []
    
    for y, x, c in coords:
        val = int(image[y, x, c])
        bit = val & 1
        bits.append(bit)
        current_byte_bits.append(bit)
        
        if len(current_byte_bits) == 8:
            byte_val = int("".join(map(str, current_byte_bits)), 2)
            if byte_val == 0:
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

def embed_dct(image: np.ndarray, message: str, channels=[0, 1, 2], Q: float = 16.0) -> np.ndarray:
    """
    Embed a text message in the quantized DCT coefficients of 8x8 blocks.
    Uses mid-frequency AC coefficients for visual imperceptibility and statistical stability.
    """
    stego_image = image.copy().astype(np.float32)
    bits = string_to_bits(message)
    total_bits = len(bits)
    
    h, w, _ = stego_image.shape
    coeff_coords = [(1, 1), (1, 2), (2, 1)]
    
    bit_idx = 0
    for y_start in range(0, h - 7, 8):
        for x_start in range(0, w - 7, 8):
            for c in channels:
                block = stego_image[y_start:y_start+8, x_start:x_start+8, c]
                dct_block = cv2.dct(block)
                
                # Quantize
                quantized = np.round(dct_block / Q).astype(np.int32)
                
                # Embed bits in the selected coefficients
                for u, v in coeff_coords:
                    if bit_idx >= total_bits:
                        dequantized = quantized.astype(np.float32) * Q
                        stego_image[y_start:y_start+8, x_start:x_start+8, c] = cv2.idct(dequantized)
                        return np.clip(stego_image, 0, 255).astype(np.uint8)
                        
                    val = int(quantized[u, v])
                    val = (val & ~1) | bits[bit_idx]
                    quantized[u, v] = val
                    bit_idx += 1
                    
                # Reconstruct block
                dequantized = quantized.astype(np.float32) * Q
                stego_image[y_start:y_start+8, x_start:x_start+8, c] = cv2.idct(dequantized)
                
    if bit_idx < total_bits:
        raise ValueError(f"Message too long! Requires {total_bits} bits, but selection only has capacity for {bit_idx} bits.")
        
    return np.clip(stego_image, 0, 255).astype(np.uint8)

def extract_dct(image: np.ndarray, channels=[0, 1, 2], Q: float = 16.0) -> str:
    """
    Extract a hidden message from the quantized DCT coefficients of 8x8 blocks.
    """
    img_float = image.astype(np.float32)
    h, w, _ = img_float.shape
    coeff_coords = [(1, 1), (1, 2), (2, 1)]
    
    bits = []
    current_byte_bits = []
    
    for y_start in range(0, h - 7, 8):
        for x_start in range(0, w - 7, 8):
            for c in channels:
                block = img_float[y_start:y_start+8, x_start:x_start+8, c]
                dct_block = cv2.dct(block)
                
                # Quantize
                quantized = np.round(dct_block / Q).astype(np.int32)
                
                for u, v in coeff_coords:
                    val = int(quantized[u, v])
                    bit = val & 1
                    bits.append(bit)
                    current_byte_bits.append(bit)
                    
                    if len(current_byte_bits) == 8:
                        byte_val = int("".join(map(str, current_byte_bits)), 2)
                        if byte_val == 0:
                            return bits_to_string(bits)
                        current_byte_bits = []
                        
    return bits_to_string(bits)

def calculate_lsb_transition_rate(image: np.ndarray, channels=[0, 1, 2]) -> float:
    """
    Calculates the spatial transition rate of the LSB bitplane.
    Clean natural images have highly correlated LSBs (transition rate < 0.465).
    Stego images with random payloads have randomized LSBs (transition rate near 0.50).
    """
    h, w, _ = image.shape
    transitions = 0
    total_pairs = 0
    
    # Check horizontal and vertical adjacent pairs in selected channels
    for c in channels:
        # Horizontal transitions
        diff_h = np.abs((image[:, :-1, c].astype(np.int32) & 1) - (image[:, 1:, c].astype(np.int32) & 1))
        transitions += np.sum(diff_h)
        total_pairs += diff_h.size
        
        # Vertical transitions
        diff_v = np.abs((image[:-1, :, c].astype(np.int32) & 1) - (image[1:, :, c].astype(np.int32) & 1))
        transitions += np.sum(diff_v)
        total_pairs += diff_v.size
        
    return float(transitions / total_pairs) if total_pairs > 0 else 0.5

def check_for_hidden_message(img: np.ndarray, keys=[42], channels=[0, 1, 2]) -> bool:
    """
    Checks if the image contains a readable ASCII message embedded using key/seed.
    """
    try:
        h, w, _ = img.shape
        total_pixels = h * w
        
        for key in keys:
            # Replicate getShuffledIndices using LCG
            indices = list(range(total_pixels))
            state = key
            for i in range(total_pixels - 1, 0, -1):
                state = (1664525 * state + 1013904223) % 4294967296
                j = state % (i + 1)
                indices[i], indices[j] = indices[j], indices[i]
                
            # Now extract bits
            current_byte_bits = []
            chars_decoded = 0
            readable_chars = 0
            
            for idx in indices:
                y = idx // w
                x = idx % w
                for c in channels:
                    bit = int(img[y, x, c]) & 1
                    current_byte_bits.append(bit)
                    if len(current_byte_bits) == 8:
                        byte_val = int("".join(map(str, current_byte_bits)), 2)
                        if byte_val == 0:  # Null terminator
                            if chars_decoded >= 3 and readable_chars == chars_decoded:
                                return True
                            break
                        
                        # Printable ASCII characters (32 to 126) + whitespace (9, 10, 13)
                        if (32 <= byte_val <= 126) or byte_val in [9, 10, 13]:
                            readable_chars += 1
                        chars_decoded += 1
                        current_byte_bits = []
                        
                        # Stop if it's garbage (non-printable chars decoded) to save time
                        if readable_chars < chars_decoded:
                            break
                            
                        if chars_decoded > 200:
                            break
                if readable_chars < chars_decoded or chars_decoded > 200:
                    break
    except Exception as e:
        print(f"Active extraction check failed: {e}")
        
    return False
