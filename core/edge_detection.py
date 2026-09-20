import cv2
import numpy as np


# ------------------ Sobel Edge Detection (Optimized) ------------------
def sobel_edge_optimized(img_gray, threshold=80, smooth=True):
    """
    Sobel Edge Detection optimized for steganography.
    - Gaussian smoothing
    - Sobel gradient magnitude
    - Normalization 0–255
    - Thresholding
    - Morphological CLOSE for stable edge regions
    """

    # 1. Optional smoothing
    if smooth:
        img = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    else:
        img = img_gray

    # 2. Sobel gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Gradient magnitude
    grad_mag = np.hypot(grad_x, grad_y)

    # 4. Normalize to [0..255]
    max_val = grad_mag.max()
    if max_val > 0:
        grad_mag = (grad_mag / max_val) * 255
    else:
        grad_mag = np.zeros_like(grad_mag)

    grad_mag = grad_mag.astype(np.uint8)

    # 5. Threshold → keep only strong edges
    grad_mag[grad_mag < threshold] = 0

    # 6. Morphological close → stabilize edges
    kernel = np.ones((3, 3), np.uint8)
    grad_mag = cv2.morphologyEx(grad_mag, cv2.MORPH_CLOSE, kernel)

    return grad_mag


# ------------------ Fuzzy Mamdani Edge Detection ------------------
def gaussian_membership(x, center, sigma):
    sigma = np.maximum(sigma, 1e-8)
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def fuzzy_mamdani_edge(img_gray):
    img_f = img_gray.astype(np.float32)
    img_f = cv2.GaussianBlur(img_f, (3, 3), 0)

    padded = np.pad(img_f, ((1, 1), (1, 1)), mode='edge')

    P1 = padded[0:-2, 0:-2]
    P2 = padded[0:-2, 1:-1]
    P3 = padded[0:-2, 2:]
    P4 = padded[1:-1, 0:-2]
    P5 = padded[1:-1, 1:-1]
    P6 = padded[1:-1, 2:]
    P7 = padded[2:, 0:-2]
    P8 = padded[2:, 1:-1]
    P9 = padded[2:, 2:]

    D1 = np.sqrt((P5 - P2)**2 + (P5 - P8)**2)
    D2 = np.sqrt((P5 - P4)**2 + (P5 - P6)**2)
    D3 = np.sqrt((P5 - P1)**2 + (P5 - P9)**2)
    D4 = np.sqrt((P5 - P3)**2 + (P5 - P7)**2)

    D_stack = np.stack([D1, D2, D3, D4], axis=0)

    Low = np.min(D_stack)
    High = np.max(D_stack)
    Medium = Low + (High - Low) / 2.0

    sigma = (High - Low) / 8.0
    sigma = np.maximum(sigma, 1e-8)

    muLow = gaussian_membership(D_stack, Low, sigma)
    muMed = gaussian_membership(D_stack, Medium, sigma)
    muHigh = gaussian_membership(D_stack, High, sigma)

    rule_edge_strong = np.max(muHigh, axis=0)
    rule_edge_medium = np.max(muMed, axis=0)
    rule_background = np.min(muLow, axis=0)

    edge_strength = np.maximum(rule_edge_strong, rule_edge_medium)
    background_strength = rule_background

    edge_binary = np.where(edge_strength > background_strength, 255, 0).astype(np.uint8)
    return edge_binary


# ------------------ Canny ------------------
def canny_with_blur(img_gray, blur_ksize=(5, 5), sigma=1.2, th1=80, th2=160):
    blurred = cv2.GaussianBlur(img_gray, blur_ksize, sigma)
    edges = cv2.Canny(blurred, th1, th2)
    return edges


# ------------------ Hybrid ------------------
def hybrid_fuzzy_or_canny(img_gray, canny_params=None):
    if canny_params is None:
        canny_params = {'blur_ksize': (5, 5), 'sigma': 1.2, 'th1': 80, 'th2': 160}

    fuzzy_map = fuzzy_mamdani_edge(img_gray)
    canny_map = canny_with_blur(
        img_gray,
        blur_ksize=canny_params['blur_ksize'],
        sigma=canny_params['sigma'],
        th1=canny_params['th1'],
        th2=canny_params['th2']
    )

    hybrid_or = cv2.bitwise_or(fuzzy_map, canny_map)
    return fuzzy_map, canny_map, hybrid_or
