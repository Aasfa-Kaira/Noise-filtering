import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Huffman Decoding
def huffman_decoding(encoded_data, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded_image = []
    current_code = ''
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_image.append(reverse_codes[current_code])
            current_code = ''
    return np.array(decoded_image, dtype=np.uint8)

# LZ77 Decompression
def lz77_decompress(compressed_data):
    decompressed_data = []
    for offset, length, next_symbol in compressed_data:
        if length > 0:
            start = len(decompressed_data) - offset
            for i in range(length):
                decompressed_data.append(decompressed_data[start + i])
        decompressed_data.append(next_symbol)
    return np.array(decompressed_data, dtype=np.uint8)

# Load Compressed Data
def load_compressed_data(filename):
    with open(filename, 'rb') as f:
        lz77_encoded, huffman_codes, original_shape = pickle.load(f)
    return lz77_encoded, huffman_codes, original_shape

def calculate_compression_ratio(original_image_path, compressed_data_path):
    original_size = os.path.getsize(original_image_path)  # Size of the original image (in bytes)
    compressed_size = os.path.getsize(compressed_data_path)  # Size of the compressed data file (in bytes)
    compression_ratio = original_size / compressed_size
    print(f"Original Image Size: {original_size} bytes")
    print(f"Compressed Data Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")

def decompress_image():
    start_time = time.time()  # Start timing

    lz77_encoded, huffman_codes, original_shape = load_compressed_data("compressed_data.pkl")
    
    # Reconstruct the Huffman encoded binary string from LZ77 symbols
    encoded_data = ''.join([huffman_codes[symbol] for _, _, symbol in lz77_encoded])
    
    # Perform Huffman decoding
    decoded_huffman = huffman_decoding(encoded_data, huffman_codes)
    
    # Decompress the LZ77 data
    decompressed_data = lz77_decompress(lz77_encoded)

    # Reshape the decompressed data to the original image shape
    decompressed_image = decompressed_data[:np.prod(original_shape)].reshape(original_shape)

    # Save decompressed image
    imageio.imwrite("decompressed_image.png", decompressed_image)

    print("Decompression Complete. Image saved as 'decompressed_image.png'.")
    calculate_compression_ratio("final_image0.1.png", "decompressed_image.png")

    # End timing and print total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"Total Decompression Time: {total_runtime:.2f} seconds")
    
    # Plotting the decompressed image
    plt.imshow(decompressed_image, cmap='gray' if decompressed_image.ndim == 2 else None)
    plt.axis('off')  # Hide axis
    plt.show()

# Example Usage
if __name__ == "__main__":
    decompress_image()
