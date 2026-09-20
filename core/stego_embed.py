import cv2
import numpy as np
import math

# ============ EDGE MODE MAP ============
EDGE_MODE_MAP = {
    "sobel": 0b01,
    "canny": 0b10,
    "hybrid": 0b11
}

def edge_mode_to_bits(mode_name: str):
    """Chuyển tên thuật toán thành 2 bits"""
    if mode_name.lower() not in EDGE_MODE_MAP:
        raise ValueError(f"Thuật toán không hợp lệ: {mode_name}")
    val = EDGE_MODE_MAP[mode_name.lower()]
    return [(val >> 1) & 1, val & 1]  # MSB first: [bit1, bit0]

def bits_to_edge_mode(bits):
    """Chuyển 2 bits thành tên thuật toán"""
    val = (bits[0] << 1) | bits[1]
    for name, code in EDGE_MODE_MAP.items():
        if code == val:
            return name
    raise ValueError(f"Edge mode không hợp lệ: {bits}")


# ============ BIT CONVERSION ============
def int_to_bits(n: int, bit_length: int):
    """Chuyển số nguyên thành list bits - MSB first"""
    return [int(b) for b in format(n, f"0{bit_length}b")]

def bits_to_int(bits):
    """Chuyển list bits (MSB first) thành số nguyên"""
    return int("".join(str(b) for b in bits), 2)

def text_to_bits(text: str):
    """Chuyển text thành list bits"""
    return [int(b) for c in text for b in format(ord(c), "08b")]

def bits_to_text(bits):
    """Chuyển list bits thành text"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(bits_to_int(byte)))
    return "".join(chars)


# ============ HEADER EMBEDDING (LSB Thuần) ============
HEADER_PIXELS = 12  # 12 pixels × 3 bits = 36 bits

def embed_header(img, edge_mode, message_length):
    """
    Nhúng header vào 12 pixels đầu tiên (LSB position 0 của mỗi channel)
    
    Header format (36 bits):
        - EDGE_MODE: 2 bits
        - MESSAGE_LENGTH: 32 bits
        - RESERVED: 2 bits (padding)
    
    Args:
        img: numpy array (H, W, 3)
        edge_mode: "sobel"/"canny"/"hybrid"
        message_length: số bits của message
    
    Returns:
        img đã nhúng header (modified in-place)
    """
    # Tạo header bits
    mode_bits = edge_mode_to_bits(edge_mode)
    length_bits = int_to_bits(message_length, 32)
    reserved_bits = [0, 0]
    
    header_bits = mode_bits + length_bits + reserved_bits  # 36 bits
    
    print(f"[HEADER] Edge mode: {edge_mode} → bits: {mode_bits}")
    print(f"[HEADER] Message length: {message_length} bits")
    print(f"[HEADER] Header bits (total 36): {header_bits[:10]}...")
    
    # Nhúng vào 12 pixels đầu (mỗi pixel 3 channels × 1 bit = 3 bits)
    h, w = img.shape[:2]
    bit_idx = 0
    
    for pixel_idx in range(HEADER_PIXELS):
        y = pixel_idx // w
        x = pixel_idx % w
        pixel = img[y, x].copy()
        
        # Nhúng 1 bit vào LSB position 0 của mỗi channel
        for ch in range(3):
            if bit_idx < len(header_bits):
                # ✅ FIX: Clear LSB đúng cách (& 0xFE tương đương & ~1 & 0xFF)
                pixel[ch] = (pixel[ch] & 0xFE) | header_bits[bit_idx]
                bit_idx += 1
        
        img[y, x] = pixel
    
    print(f"[HEADER] ✅ Đã nhúng {bit_idx} bits header vào {HEADER_PIXELS} pixels")
    return img


def extract_header(img):
    """
    Trích xuất header từ 12 pixels đầu
    
    Returns:
        (edge_mode, message_length)
    """
    print("[EXTRACT HEADER] Đọc 36 bits từ 12 pixels đầu...")
    
    h, w = img.shape[:2]
    header_bits = []
    
    for pixel_idx in range(HEADER_PIXELS):
        y = pixel_idx // w
        x = pixel_idx % w
        pixel = img[y, x]
        
        # Đọc LSB position 0 của mỗi channel
        for ch in range(3):
            header_bits.append(int(pixel[ch]) & 1)
    
    print(f"[EXTRACT HEADER] Header bits: {header_bits[:10]}...")
    
    # Parse header
    mode_bits = header_bits[0:2]
    length_bits = header_bits[2:34]
    
    edge_mode = bits_to_edge_mode(mode_bits)
    message_length = bits_to_int(length_bits)
    
    print(f"[EXTRACT HEADER] ✅ Edge mode: {edge_mode}")
    print(f"[EXTRACT HEADER] ✅ Message length: {message_length} bits")
    
    return edge_mode, message_length


# ============ DATA EMBEDDING (Self-Describing với Flag) ============
def embed_data_with_flags(img, edge_map, message_bits):
    """
    Nhúng message data từ pixel 12 trở đi, mỗi pixel có flag
    
    Edge pixel (10 bits/pixel):
        R: [flag=1][data2][data1][data0] (4 LSBs)
        G: [data5][data4][data3] (3 LSBs)
        B: [data8][data7][data6] (3 LSBs)
    
    Non-edge pixel (4 bits/pixel):
        R: [flag=0][data0] (2 LSBs)
        G: [data1] (1 LSB)
        B: [data2] (1 LSB)
    
    Args:
        img: ảnh đã có header
        edge_map: edge map
        message_bits: list bits cần nhúng
    
    Returns:
        stego_img, bits_embedded
    """
    stego = img.copy()
    h, w = stego.shape[:2]
    
    # ✅ FIX: Kiểm tra edge_map size
    if edge_map.shape[:2] != (h, w):
        raise ValueError(
            f"❌ Edge map không khớp kích thước ảnh!\n"
            f"Ảnh: (H={h}, W={w})\n"
            f"Edge map: (H={edge_map.shape[0]}, W={edge_map.shape[1]})\n"
            f"Hai ảnh phải có cùng kích thước!"
        )
    
    # Calculate capacity
    total_pixels = h * w
    data_pixels = total_pixels - HEADER_PIXELS
    
    edge_pixels = 0
    non_edge_pixels = 0
    
    for pixel_idx in range(HEADER_PIXELS, total_pixels):
        y = pixel_idx // w
        x = pixel_idx % w
        if edge_map[y, x] > 0:
            edge_pixels += 1
        else:
            non_edge_pixels += 1
    
    capacity = edge_pixels * 9 + non_edge_pixels * 3  # bits (không tính flag)
    
    print(f"[DATA] Pixels available: {data_pixels}")
    print(f"[DATA] Edge pixels: {edge_pixels} (9 bits data each)")
    print(f"[DATA] Non-edge pixels: {non_edge_pixels} (3 bits data each)")
    print(f"[DATA] Capacity: {capacity} bits")
    print(f"[DATA] Message: {len(message_bits)} bits")
    
    if len(message_bits) > capacity:
        raise ValueError(
            f"Message quá lớn!\n"
            f"Cần: {len(message_bits)} bits\n"
            f"Capacity: {capacity} bits\n"
            f"Vui lòng rút ngắn message hoặc dùng ảnh lớn hơn."
        )
    
    # Embed data
    bit_idx = 0
    pixels_used = 0
    
    for pixel_idx in range(HEADER_PIXELS, total_pixels):
        if bit_idx >= len(message_bits):
            break
        
        y = pixel_idx // w
        x = pixel_idx % w
        pixel = stego[y, x].copy()
        is_edge = edge_map[y, x] > 0
        
        if is_edge:
            # Edge pixel: 10 bits total (1 flag + 9 data)
            # R channel: [flag=1][d2][d1][d0] (bits 3-2-1-0)
            flag_and_data = [1]  # flag
            for _ in range(3):
                if bit_idx < len(message_bits):
                    flag_and_data.append(message_bits[bit_idx])
                    bit_idx += 1
                else:
                    flag_and_data.append(0)
            
            # ✅ FIX: Nhúng vào R (4 LSBs) với mask 8-bit
            for i, bit in enumerate(flag_and_data):
                mask = (0xFF ^ (1 << i))  # Clear bit i
                pixel[2] = (int(pixel[2]) & mask) | (bit << i)
            
            # G channel: [d5][d4][d3] (bits 2-1-0)
            for i in range(3):
                if bit_idx < len(message_bits):
                    bit = message_bits[bit_idx]
                    bit_idx += 1
                else:
                    bit = 0
                mask = (0xFF ^ (1 << i))
                pixel[1] = (int(pixel[1]) & mask) | (bit << i)
            
            # B channel: [d8][d7][d6] (bits 2-1-0)
            for i in range(3):
                if bit_idx < len(message_bits):
                    bit = message_bits[bit_idx]
                    bit_idx += 1
                else:
                    bit = 0
                mask = (0xFF ^ (1 << i))
                pixel[0] = (int(pixel[0]) & mask) | (bit << i)
        
        else:
            # Non-edge pixel: 4 bits total (1 flag + 3 data)
            # R channel: [flag=0][d0] (bits 1-0)
            flag_and_data = [0]  # flag
            if bit_idx < len(message_bits):
                flag_and_data.append(message_bits[bit_idx])
                bit_idx += 1
            else:
                flag_and_data.append(0)
            
            # ✅ FIX: Nhúng với mask 8-bit
            for i, bit in enumerate(flag_and_data):
                mask = (0xFF ^ (1 << i))
                pixel[2] = (int(pixel[2]) & mask) | (bit << i)
            
            # G channel: [d1] (bit 0)
            if bit_idx < len(message_bits):
                bit = message_bits[bit_idx]
                bit_idx += 1
            else:
                bit = 0
            pixel[1] = (int(pixel[1]) & 0xFE) | bit  # Clear bit 0
            
            # B channel: [d2] (bit 0)
            if bit_idx < len(message_bits):
                bit = message_bits[bit_idx]
                bit_idx += 1
            else:
                bit = 0
            pixel[0] = (int(pixel[0]) & 0xFE) | bit  # Clear bit 0
        
        stego[y, x] = pixel
        pixels_used += 1
    
    print(f"[DATA] ✅ Đã nhúng {bit_idx} bits data vào {pixels_used} pixels")
    
    return stego, bit_idx, edge_pixels, non_edge_pixels



def extract_data_with_flags(img, message_length):
    """
    Trích xuất data từ pixel 12 trở đi, tự động phát hiện flag
    
    Returns:
        message_bits (list)
    """
    print(f"[EXTRACT DATA] Cần đọc {message_length} bits...")
    
    h, w = img.shape[:2]
    total_pixels = h * w
    message_bits = []
    pixels_read = 0
    edge_count = 0
    non_edge_count = 0
    
    for pixel_idx in range(HEADER_PIXELS, total_pixels):
        if len(message_bits) >= message_length:
            break
        
        y = pixel_idx // w
        x = pixel_idx % w
        pixel = img[y, x]
        
        # Đọc flag từ R channel LSB position 0
        flag = int(pixel[2]) & 1
        
        if flag == 1:
            # Edge pixel: đọc 9 bits data
            edge_count += 1
            
            # R: bits 3-2-1 (bỏ bit 0 là flag)
            for i in range(1, 4):
                if len(message_bits) < message_length:
                    message_bits.append((int(pixel[2]) >> i) & 1)
            
            # G: bits 2-1-0
            for i in range(3):
                if len(message_bits) < message_length:
                    message_bits.append((int(pixel[1]) >> i) & 1)
            
            # B: bits 2-1-0
            for i in range(3):
                if len(message_bits) < message_length:
                    message_bits.append((int(pixel[0]) >> i) & 1)
        
        else:
            # Non-edge pixel: đọc 3 bits data
            non_edge_count += 1
            
            # R: bit 1 (bỏ bit 0 là flag)
            if len(message_bits) < message_length:
                message_bits.append((int(pixel[2]) >> 1) & 1)
            
            # G: bit 0
            if len(message_bits) < message_length:
                message_bits.append(int(pixel[1]) & 1)
            
            # B: bit 0
            if len(message_bits) < message_length:
                message_bits.append(int(pixel[0]) & 1)
        
        pixels_read += 1
    
    print(f"[EXTRACT DATA] ✅ Đã đọc {len(message_bits)} bits từ {pixels_read} pixels")
    print(f"[EXTRACT DATA] Edge pixels: {edge_count}, Non-edge: {non_edge_count}")
    
    return message_bits[:message_length]


# ============ WRAPPER FUNCTIONS ============
def embed_message(cover_img_path, edge_map, secret_message, output_path, algorithm, x=None, y=None):
    """
    Hàm nhúng message - giữ tương thích với code cũ (x, y không dùng nữa)
    
    Args:
        cover_img_path: đường dẫn ảnh gốc hoặc numpy array
        edge_map: edge map (numpy array)
        secret_message: string cần nhúng
        output_path: đường dẫn lưu ảnh stego
        algorithm: "sobel"/"canny"/"hybrid"
        x, y: (deprecated) giữ để tương thích, không dùng nữa
    """
    print("\n" + "="*60)
    print("BẮT ĐẦU NHÚNG MESSAGE")
    print("="*60)
    
    # Đọc ảnh
    if isinstance(cover_img_path, str):
        img = cv2.imread(cover_img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {cover_img_path}")
    else:
        img = cover_img_path.copy()
    
    # ⭐ LƯU ẢNH GỐC CHƯA BỊ CAN THIỆP (để tính PSNR/SSIM/MSE đúng)
    original_img = img.copy()

    
    # ✅ Validate input
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Ảnh phải là RGB/BGR 3 channels, nhận được: {img.shape}")
    
    print(f"[1] Ảnh: {img.shape}")
    
    # Chuyển message thành bits
    message_bits = text_to_bits(secret_message)
    message_length = len(message_bits)
    
    print(f"[2] Message: '{secret_message[:30]}...'")
    print(f"    Length: {message_length} bits ({message_length//8} bytes)")
    
    # Nhúng header
    img = embed_header(img, algorithm, message_length)
    
    # Nhúng data
    stego_img, bits_embedded, edge_pixels, non_edge_pixels = \
    embed_data_with_flags(img, edge_map, message_bits)

    
    # Lưu ảnh
    success = cv2.imwrite(output_path, stego_img)
    if not success:
        raise IOError(f"Không thể lưu ảnh stego: {output_path}")
    
    print(f"[3] ✅ Đã lưu: {output_path}")
    print(f"    Total bits: {36 + bits_embedded} (36 header + {bits_embedded} data)")
    print("="*60 + "\n")
    
    # === COMPUTE METRICS ===

    metrics = compute_all_metrics(
        original_img=original_img,   # ảnh gốc thật
        stego_img=stego_img,
        edge_pixels=edge_pixels,
        non_edge_pixels=non_edge_pixels,
        bits_embedded=bits_embedded
)

    return metrics


def extract_message(stego_img_path, edge_map=None, expected_algorithm=None, x=None, y=None):
    """
    Hàm trích xuất message - KHÔNG CẦN edge_map, x, y nữa
    
    Args:
        stego_img_path: đường dẫn ảnh stego hoặc numpy array
        edge_map: (deprecated) không dùng nữa, giữ để tương thích
        expected_algorithm: (optional) để kiểm tra, nếu None thì tự động lấy từ header
        x, y: (deprecated) không dùng nữa
        
    Returns:
        tuple: (message, algorithm_used)
        - message: string đã trích xuất
        - algorithm_used: thuật toán đã dùng khi nhúng
    """
    print("\n" + "="*60)
    print("BẮT ĐẦU TRÍCH XUẤT MESSAGE")
    print("="*60)
    
    # Đọc ảnh
    if isinstance(stego_img_path, str):
        img = cv2.imread(stego_img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh stego: {stego_img_path}")
    else:
        img = stego_img_path.copy()
    
    # ✅ Validate input
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Ảnh phải là RGB/BGR 3 channels, nhận được: {img.shape}")
    
    print(f"[1] Ảnh stego: {img.shape}")
    
    # Trích xuất header
    try:
        edge_mode, message_length = extract_header(img)
    except Exception as e:
        raise ValueError(
            f"❌ Không thể đọc header từ ảnh!\n"
            f"Ảnh này có thể:\n"
            f"  - Không phải ảnh stego của chương trình này\n"
            f"  - Đã bị chỉnh sửa/nén sau khi nhúng\n"
            f"Chi tiết lỗi: {str(e)}"
        )
    
    # Kiểm tra algorithm nếu được chỉ định
    if expected_algorithm:
        if edge_mode.lower() != expected_algorithm.lower():
            raise ValueError(
                f"❌ SAI THUẬT TOÁN!\n\n"
                f"Thuật toán đã dùng khi nhúng: {edge_mode.upper()}\n"
                f"Thuật toán bạn đang chọn: {expected_algorithm.upper()}\n\n"
                f"⚠️  Vui lòng:\n"
                f"  1. Chọn đúng thuật toán '{edge_mode.upper()}', hoặc\n"
                f"  2. Để trống (tự động phát hiện)"
            )
        print(f"[2] ✅ Algorithm khớp: {edge_mode.upper()}")
    else:
        print(f"[2] 🔍 Tự động phát hiện thuật toán: {edge_mode.upper()}")
    
    # Kiểm tra message length hợp lệ
    if message_length <= 0 or message_length > 10_000_000:  # max 10MB text
        raise ValueError(
            f"❌ Độ dài message không hợp lệ: {message_length} bits\n"
            f"Có thể ảnh không phải stego image hoặc đã bị hỏng."
        )
    
    # Trích xuất data (tự động dùng flag, không cần edge map)
    message_bits = extract_data_with_flags(img, message_length)
    
    # Chuyển về text
    try:
        message = bits_to_text(message_bits)
    except Exception as e:
        raise ValueError(
            f"❌ Không thể decode message từ bits!\n"
            f"Data có thể bị hỏng hoặc không đúng định dạng.\n"
            f"Chi tiết: {str(e)}"
        )
    
    print(f"[3] ✅ Message: '{message[:50]}...'")
    print(f"    Length: {len(message)} characters")
    print("="*60 + "\n")
    
    return message, edge_mode



# ============ METRICS ============
class Metrics:
    def __init__(self, psnr, ssim, mse,
                 embedded_bits, capacity_bits, percent_used,
                 bpp, edge_pixels, non_edge_pixels):
        self.psnr = psnr
        self.ssim = ssim
        self.mse = mse
        self.embedded_bits = embedded_bits
        self.capacity_bits = capacity_bits
        self.percent_used = percent_used
        self.bpp = bpp
        self.edge_pixels = edge_pixels
        self.non_edge_pixels = non_edge_pixels

def compute_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def compute_all_metrics(original_img, stego_img,
                        edge_pixels, non_edge_pixels,
                        bits_embedded):

    # MSE
    diff = original_img.astype(np.float64) - stego_img.astype(np.float64)
    mse = np.mean(diff ** 2)

    # PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)

    # SSIM
    ssim = compute_ssim(original_img, stego_img)

    # TOTAL CAPACITY
    capacity_bits = edge_pixels * 9 + non_edge_pixels * 3

    # percent
    percent_used = (bits_embedded / capacity_bits) * 100 if capacity_bits > 0 else 0

    # BPP
    H, W = original_img.shape[:2]
    bpp = bits_embedded / (H * W)

    return Metrics(
        psnr=psnr,
        ssim=ssim,
        mse=mse,
        embedded_bits=bits_embedded,
        capacity_bits=capacity_bits,
        percent_used=percent_used,
        bpp=bpp,
        edge_pixels=edge_pixels,
        non_edge_pixels=non_edge_pixels
    )

