
import librosa
import librosa.display
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os

def perform_wavelet_analysis(file_path):
    # 1. Load the signal at 8kHz 
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    signal, sr = librosa.load(file_path, sr=8000)
    
    # 2. Discrete Wavelet Transform (DWT) Decomposition
    # Using 'db4' (Daubechies 4) 
    # Decomposition level = 3 (results in cA3, cD3, cD2, cD1)
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    
    # 3. Feature Extraction (Energy and Entropy)
    print("--- Wavelet Feature Extraction ---")
    features = []
    for i, c in enumerate(coeffs):
        level_name = "Approx (cA3)" if i == 0 else f"Detail (cD{4-i})"
        
        # Calculate Energy (Sum of squares)
        energy = np.sum(np.square(c))
        
        # Calculate Shannon Entropy
        # Normalize coefficients to represent a probability distribution
        prob = np.square(c) / (energy + 1e-9)
        entropy = -np.sum(prob * np.log(prob + 1e-9))
        
        features.append((energy, entropy))
        print(f"{level_name}: Energy = {energy:.4f}, Entropy = {entropy:.4f}")

    # 4. Visualization: Time-Domain vs. Scalogram
    plt.figure(figsize=(12, 8))

    # Plot 1: Time Domain
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(signal, sr=sr, color='blue')
    plt.title(f"Time-Domain Signal: {os.path.basename(file_path)}")
    plt.ylabel("Amplitude")

    # Plot 2: Scalogram (CWT for visualization)
    plt.subplot(2, 1, 2)
    widths = np.arange(1, 64)
    cwtmatr, freqs = pywt.cwt(signal, widths, 'mexh') # Mexican Hat wavelet
    plt.imshow(np.abs(cwtmatr), extent=[0, len(signal)/sr, 1, 64], 
               cmap='inferno', aspect='auto', interpolation='nearest')
    plt.title("Wavelet Scalogram (Time-Frequency Analysis)")
    plt.ylabel("Scale")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Magnitude")

    plt.tight_layout()
    plt.show()

# --- RUN THE ANALYSIS ---
test_file = os.path.join("recordings", "0_jackson_0.wav")
perform_wavelet_analysis(test_file)