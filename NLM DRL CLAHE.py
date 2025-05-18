import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import jit  

# Load the original image
img = plt.imread(r"C:\Users\Admin\Desktop\projects\digital image processing\DIP denoising python codes\ncccmusk.jpg")  


def manual_clahe(img, clip_limit=0.001, tile_grid_size=(8, 8)):
    if len(img.shape) == 3:  # Color image
        channels = cv2.split(img)
        clahe_channels = [] 
        
        for ch in channels:
            ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            clahe_ch = clahe.apply(ch)
            clahe_channels.append(clahe_ch)
        
        clahe_img = cv2.merge(clahe_channels)
    
    else:  # Grayscale image
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_img = clahe.apply(img)

    return clahe_img

@jit(nopython=True, fastmath=True)
def nlm_denoise_with_jit(img, search_w, sim_w, h):
    img = img.astype(np.float32)
    hgt, wdt, ch = img.shape
    denoised = np.zeros_like(img)

    for i in range(hgt):
        for j in range(wdt):
            i_min, i_max = max(0, i - search_w // 2), min(hgt, i + search_w // 2 + 1)
            j_min, j_max = max(0, j - search_w // 2), min(wdt, j + search_w // 2 + 1)
            sw = img[i_min:i_max, j_min:j_max, :]

            w = np.exp(-((sw - img[i, j, :]) ** 2).sum(axis=2) / (h ** 2))
            w /= w.sum()

            for c in range(ch):
                denoised[i, j, c] = (w * sw[:, :, c]).sum()

    return denoised

@jit(nopython=True, fastmath=True)
def mse(image1, image2):
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

@jit(nopython=True, fastmath=True)
def psnr(image1, image2):
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse_val)

@jit(nopython=True, fastmath=True)
def mean_manual(image):
    total_sum = np.sum(image)
    num_elements = image.size
    return total_sum / num_elements

@jit(nopython=True, fastmath=True)
def ssim(image1, image2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = mean_manual(image1.astype(np.float32))
    mu2 = mean_manual(image2.astype(np.float32))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = mean_manual(image1.astype(np.float32) ** 2) - mu1_sq
    sigma2_sq = mean_manual(image2.astype(np.float32) ** 2) - mu2_sq
    sigma12 = mean_manual(image1.astype(np.float32) * image2.astype(np.float32)) - mu1_mu2

    ssim_value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                 ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_value

@jit(nopython=True, fastmath=True)
def fsim(image1, image2):
    C = 0.01
    gradient1 = np.sqrt(image1.astype(np.float32) ** 2)
    gradient2 = np.sqrt(image2.astype(np.float32) ** 2)

    num = 2 * gradient1 * gradient2 + C
    den = gradient1**2 + gradient2**2 + C
    FSIM_map = num / den

    return np.mean(FSIM_map)

# Define parameter ranges for NLM
search_w_range = range(3, 31, 4)  
sim_w_range = range(3, 17, 4)      
h_range = range(37, 63, 6)         

all_metrics = [] 
start_time = time.time()

for search_w in search_w_range:
    for sim_w in sim_w_range:
        for h in h_range:
            denoised_img = nlm_denoise_with_jit(img, search_w, sim_w, h)
            psnr_val = psnr(img, denoised_img)
            ssim_val = ssim(img, denoised_img)
            fsim_val = fsim(img, denoised_img)
            mse_val = mse(img, denoised_img)


            all_metrics.append({
                'search_w': search_w,
                'sim_w': sim_w,
                'h': h,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'fsim': fsim_val,
                'mse': mse_val,
            })
for search_w in search_w_range:
    for sim_w in sim_w_range:
        for h in h_range:
            # Apply NLM denoising with current parameters
            denoised_img = nlm_denoise_with_jit(img, search_w, sim_w, h) 

            # Calculate metrics for the entire image
            psnr_val = psnr(img, denoised_img)
            ssim_val = ssim(img, denoised_img)
            fsim_val = fsim(img, denoised_img)
            mse_val = mse(img, denoised_img)

        

            # Store the metrics for the current combination of parameters
            all_metrics.append({
                'search_w': search_w,
                'sim_w': sim_w,
                'h': h,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'fsim': fsim_val,
                'mse': mse_val,
                # MSE for the cropped area
            })

# Calculate combined reward for each set of metrics
best_combined_reward = -float('inf')
best_params = None

for metrics in all_metrics:
    combined_reward = (0.5 * metrics['psnr'] + 0.3 * metrics['mse'] + 0.3 * metrics['ssim'] + 0.3 * metrics['fsim'])  # Weighted reward with more focus on PSNR
    if combined_reward > best_combined_reward:
        best_combined_reward = combined_reward
        best_params = metrics
        
# Now, best_params will contain the parameters with the highest combined reward using normalized metrics


def calculate_histogram(image_array):
    if len(image_array.shape) == 3:
        image_array = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    histogram = [0] * 256
    for pixel_value in image_array.ravel():
        histogram[pixel_value] += 1

    return histogram


psnrf_val = psnr(img, denoised_img)
ssimf_val = ssim(img, denoised_img)
fsimf_val = fsim(img, denoised_img)
mse_val = mse(img, denoised_img)

print("Best Parameters based on Combined Reward:")
print(f"Search Window: {best_params['search_w']}")
print(f"Similarity Window: {best_params['sim_w']}")
print(f"h: {best_params['h']}\n")

print("Metrics for Best Parameters:")
print(f"PSNR: {best_params['psnr']}")
print(f"SSIM: {best_params['ssim']}")
print(f"FSIM: {best_params['fsim']}")
print(f"MSE : {best_params['mse']}")


best_denoised_img = nlm_denoise_with_jit(img, best_params['search_w'], best_params['sim_w'], best_params['h'])
best_denoised_img_display = np.clip(best_denoised_img, 0, 255).astype(np.uint8)

# Display original image
plt.figure(figsize=(10, 5))
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Display denoised image
plt.figure(figsize=(10, 5))
plt.imshow(best_denoised_img.astype(np.uint8))
plt.title('Denoised Image')
plt.axis('off')
plt.show()

# Apply CLAHE to the denoised image
clahe_img = manual_clahe(best_denoised_img_display, clip_limit=2 , tile_grid_size=(13, 13))
final_img = nlm_denoise_with_jit(clahe_img, 21, 3, 31)


# Display CLAHE image
plt.figure(figsize=(10, 5))
plt.imshow(clahe_img.astype(np.uint8))
plt.title('CLAHE Image')
plt.axis('off')
plt.show()

# Display CLAHE image
plt.figure(figsize=(10, 5))
plt.imshow(final_img.astype(np.uint8))
plt.title('final Image')
plt.axis('off')
plt.show()


#plt.imsave('final_image#.png', final_img.astype(np.uint8))
#plt.imsave('denoised_image.png', best_denoised_img.astype(np.uint8))
#plt.imsave('clahe_image#.png', clahe_img.astype(np.uint8))



# Calculate histograms
original_hist = calculate_histogram(img)
denoised_hist = calculate_histogram(best_denoised_img_display)
clahed_hist = calculate_histogram(clahe_img)


# Histogram of Original Image
plt.figure(figsize=(10, 5))
plt.title('Histogram of Original Image', fontsize=16, fontweight='bold')
plt.bar(range(256), original_hist, color='blue', alpha=0.7, width=0.7)
plt.xlabel('Pixel Value', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlim(0, 255)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram of Denoised Image
plt.figure(figsize=(10, 5))
plt.title('Histogram of Denoised Image', fontsize=16, fontweight='bold')
plt.bar(range(256), denoised_hist, color='maroon', alpha=0.7, width=0.7)
plt.xlabel('Pixel Value', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlim(0, 255)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram of CLAHE Image
plt.figure(figsize=(10, 5))
plt.title('Histogram of CLAHE Image', fontsize=16, fontweight='bold')
plt.bar(range(256), clahed_hist, color='#006633', alpha=0.7, width=0.7)
plt.xlabel('Pixel Value', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlim(0, 255)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram of Final Image
plt.figure(figsize=(10, 5))
plt.title('Histogram of Final Image', fontsize=16, fontweight='bold')
plt.bar(range(256), clahed_hist, color='black', alpha=0.7, width=0.7)
plt.xlabel('Pixel Value', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlim(0, 255)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





end_time = time.time()
print(f"Total Processing Time: {end_time - start_time} seconds")
