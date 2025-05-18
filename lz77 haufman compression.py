import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from numba import jit

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

def calculate_frequencies(image):
    freq = np.zeros(256, dtype=np.int32)
    for pixel in image.flatten():
        freq[pixel] += 1
    return freq

def build_huffman_tree(frequencies):
    nodes = [Node(symbol, frequencies[symbol]) for symbol in range(len(frequencies)) if frequencies[symbol] > 0]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        nodes.append(parent)
    return nodes[0]

def generate_huffman_codes(tree, prefix='', codebook=None):
    if codebook is None:
        codebook = {}
    if tree.symbol is not None:
        codebook[tree.symbol] = prefix
    else:
        generate_huffman_codes(tree.left, prefix + '0', codebook)
        generate_huffman_codes(tree.right, prefix + '1', codebook)
    return codebook

def huffman_compress(image):
    frequencies = calculate_frequencies(image)
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(huffman_tree)
    encoded_image = ''.join(huffman_codes[pixel] for pixel in image.flatten())
    return encoded_image, huffman_codes

def lz77_compress(data, window_size=20, lookahead_buffer_size=15):
    compressed_data = []
    i = 0
    while i < len(data):
        match = (-1, -1)
        max_match_len = 0
        for j in range(max(0, i - window_size), i):
            k = 0
            while (k < lookahead_buffer_size and i + k < len(data) and data[j + k] == data[i + k]):
                k += 1
            if k > max_match_len:
                match = (i - j, k)
                max_match_len = k
        if max_match_len > 0:
            compressed_data.append((match[0], max_match_len, data[i + max_match_len] if i + max_match_len < len(data) else 0))
            i += max_match_len + 1
        else:
            compressed_data.append((0, 0, data[i]))
            i += 1
    return compressed_data

@jit(nopython=True)
def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size if compressed_size > 0 else float('inf')

@jit(nopython=True)
def get_memory_size(size_in_bits):
    size_in_bytes = size_in_bits / 8
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_bytes, size_in_kb, size_in_mb

def save_compressed_data(filename, lz77_encoded, huffman_codes, original_shape):
    with open(filename, 'wb') as f:
        pickle.dump((lz77_encoded, huffman_codes, original_shape), f)
    print("Compression data saved to:", filename)

def compress_image(image):
    start_time = time.time()

    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Store the original image shape
    original_shape = image.shape

    # Flatten the image for compression
    flat_image = image.flatten()

    # Perform LZ77 compression
    lz77_encoded = lz77_compress(flat_image)

    # Extract the symbols from the LZ77 encoded data and perform Huffman encoding
    huffman_data = np.array([symbol for _, _, symbol in lz77_encoded], dtype=np.uint8)
    encoded_data, huffman_codes = huffman_compress(huffman_data)

    # Save the compressed data and image shape
    save_compressed_data("compressed_data.pkl", lz77_encoded, huffman_codes, original_shape)

    # Calculate compression ratio
    original_size = flat_image.nbytes * 8  # Size in bits
    compressed_size = len(encoded_data)
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)

    # Calculate memory size of the compressed data
    size_in_bytes, size_in_kb, size_in_mb = get_memory_size(compressed_size)

    total_runtime = time.time() - start_time
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Compressed Size: {size_in_bytes:.2f} bytes, {size_in_kb:.2f} KB, {size_in_mb:.2f} MB")
    print(f"Total Compression Time: {total_runtime:.2f} seconds")

# Example Usage
if __name__ == "__main__":
    image = plt.imread("final_image0.1.png")  # Update this path to your image file
    compress_image(image)
