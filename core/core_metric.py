import numpy as np
import cv2

# ------------------ PSNR ------------------
def compute_psnr(original_path, modified_path):
    """
    Tính toán PSNR giữa 2 ảnh
    """
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(modified_path)
    if img1 is None or img2 is None:
        raise ValueError("Không đọc được ảnh để so sánh.")
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr


# ------------------ Payload ------------------
def compute_payload(secret_text):
    """
    Tính payload = tổng số bit thông điệp
    """
    return len(secret_text) * 8


# ------------------ Ratio ------------------
def compute_ratio(secret_text, img_path):
    """
    Tính tỷ lệ giữa dữ liệu giấu và dung lượng ảnh
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Không đọc được ảnh.")
    payload = len(secret_text) * 8
    ratio = payload / img.size
    return ratio


# ------------------ Gói kết quả ------------------
def evaluate_stego(original_path, stego_path, secret_text):
    """
    Tính tất cả thông số PSNR, payload, ratio cùng lúc
    """
    psnr = compute_psnr(original_path, stego_path)
    payload = compute_payload(secret_text)
    ratio = compute_ratio(secret_text, original_path)
    return psnr, payload, ratio
