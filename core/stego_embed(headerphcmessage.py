import cv2
import numpy as np
import math

# ============ Algorithm ID Management ============
ALGORITHM_MAP = {
    "sobel": 1,
    "canny": 2,
    "hybrid": 3
}

def algorithm_to_id(algo_name: str) -> int:
    """Chuyển tên thuật toán thành ID (8 bits)"""
    algo_lower = algo_name.lower()
    if algo_lower not in ALGORITHM_MAP:
        raise ValueError(f"Thuật toán không hợp lệ: {algo_name}")
    return ALGORITHM_MAP[algo_lower]

def id_to_algorithm(algo_id: int) -> str:
    """Chuyển ID thành tên thuật toán"""
    for name, id_val in ALGORITHM_MAP.items():
        if id_val == algo_id:
            return name
    raise ValueError(f"Algorithm ID không hợp lệ: {algo_id}")


# ============ Fixed Position Embedding (cho algorithm_id) ============
def embed_fixed_8bits(img, bits_8):
    """
    Nhúng 8 bits vào vị trí cố định (pixel đầu tiên, LSB thông thường)
    
    Args:
        img: numpy array (H, W, 3)
        bits_8: list 8 bits [0/1, 0/1, ...] - LSB first format
    
    Returns:
        img đã nhúng (modified in-place nhưng vẫn return)
    """
    if len(bits_8) != 8:
        raise ValueError(f"Cần đúng 8 bits, nhận được {len(bits_8)}")
    
    # Nhúng vào pixel (0,0) - 3 channels
    # Blue channel: bits 0,1,2 (LSB positions 0,1,2)
    # Green channel: bits 3,4,5 (LSB positions 0,1,2)
    # Red channel: bits 6,7 (LSB positions 0,1)
    
    pixel = img[0, 0].copy()
    
    # Blue channel: bits[0,1,2] -> LSB positions 0,1,2
    for i in range(3):
        mask = 0xFF ^ (1 << i)  # Clear bit i
        pixel[0] = (pixel[0] & mask) | (int(bits_8[i]) << i)
    
    # Green channel: bits[3,4,5] -> LSB positions 0,1,2
    for i in range(3):
        mask = 0xFF ^ (1 << i)
        pixel[1] = (pixel[1] & mask) | (int(bits_8[3 + i]) << i)
    
    # Red channel: bits[6,7] -> LSB positions 0,1
    for i in range(2):
        mask = 0xFF ^ (1 << i)
        pixel[2] = (pixel[2] & mask) | (int(bits_8[6 + i]) << i)
    
    img[0, 0] = pixel
    
    print(f"[DEBUG embed_fixed_8bits] Pixel (0,0) sau nhúng: {img[0,0]}")
    print(f"[DEBUG embed_fixed_8bits] Bits nhúng: {bits_8}")
    
    return img


def extract_fixed_8bits(img):
    """
    Trích xuất 8 bits từ vị trí cố định (pixel đầu tiên)
    
    Args:
        img: numpy array (H, W, 3)
    
    Returns:
        list 8 bits [0/1, ...] - LSB first format
    """
    pixel = img[0, 0]
    bits = []
    
    print(f"[DEBUG extract_fixed_8bits] Pixel (0,0): {pixel}")
    
    # Blue channel: LSB positions 0,1,2 -> bits[0,1,2]
    for i in range(3):
        bits.append((int(pixel[0]) >> i) & 1)
    
    # Green channel: LSB positions 0,1,2 -> bits[3,4,5]
    for i in range(3):
        bits.append((int(pixel[1]) >> i) & 1)
    
    # Red channel: LSB positions 0,1 -> bits[6,7]
    for i in range(2):
        bits.append((int(pixel[2]) >> i) & 1)
    
    print(f"[DEBUG extract_fixed_8bits] Bits trích xuất: {bits}")
    
    return bits


# ---------------- Bit Conversion ----------------
def text_to_bits(text: str):
    """
    Chuyển text (ASCII / English) thành list các bit.
    Mỗi ký tự -> 8 bit (ord/chr) giống code console của bạn.
    """
    return [int(b) for c in text for b in format(ord(c), "08b")]


def int_to_bits(n: int, bit_length: int = 32):
    """Chuyển số nguyên thành list bits - MSB first"""
    return [int(b) for b in format(n, f"0{bit_length}b")]


def bits_to_int(bits):
    """Chuyển list bits (MSB first) thành số nguyên"""
    return int("".join(str(b) for b in bits), 2)


def bits_to_text(bits):
    """
    Chuyển list bit -> string (dùng chr, phù hợp text tiếng Anh).
    """
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) < 8:
            break
        chars.append(chr(bits_to_int(byte)))
    return "".join(chars)


# ---------------- LSB Embedding ----------------
def embed_mlsb(img_cover, edge_map, secret_bits, x=3, y=1):
    """
    Nhúng tin nhắn vào ảnh sử dụng MLSB với edge detection.

    KHÔNG tự tạo header trong hàm này.
    secret_bits đã bao gồm header (32 bit length) + message.

    Args:
        img_cover: ảnh gốc (BGR, uint8) hoặc đường dẫn
        edge_map: edge map (grayscale, cùng kích thước với ảnh)
        secret_bits: list bit để nhúng (header + data)
        x: số bit nhúng mỗi channel ở vùng edge (1–8)
        y: số bit nhúng mỗi channel ở vùng non-edge (1–8)

    Returns:
        stego: ảnh đã nhúng
        bits_embedded: tổng số bit đã nhúng
    """
    # Đọc ảnh nếu là đường dẫn
    if isinstance(img_cover, str):
        img_cover = cv2.imread(img_cover)
        if img_cover is None:
            raise ValueError(f"Không thể đọc ảnh: {img_cover}")

    if img_cover is None:
        raise ValueError("img_cover is None")
    if edge_map is None:
        raise ValueError("edge_map is None")

    # Validate input
    if not (1 <= x <= 8 and 1 <= y <= 8):
        raise ValueError(f"x và y phải trong khoảng 1-8, nhận được x={x}, y={y}")

    if len(img_cover.shape) != 3 or img_cover.shape[2] != 3:
        raise ValueError("Ảnh phải có 3 channels (BGR)")

    if edge_map.shape[:2] != img_cover.shape[:2]:
        raise ValueError("Edge map phải có cùng kích thước với ảnh")

    stego = img_cover.copy().astype(np.uint8)
    h, w, c = stego.shape

    # Tính capacity
    edge_pixels = np.sum(edge_map > 0)
    non_edge_pixels = h * w - edge_pixels
    capacity = edge_pixels * x * c + non_edge_pixels * y * c

    if len(secret_bits) > capacity:
        raise ValueError(
            f"Message quá lớn!\n"
            f"  - Cần: {len(secret_bits)} bits ({len(secret_bits)//8} bytes)\n"
            f"  - Capacity: {capacity} bits ({capacity//8} bytes)\n"
        )

    print(f"[*] Image capacity: {capacity} bits ({capacity//8} bytes)")
    print(
        f"[*] Message size (with header): "
        f"{len(secret_bits)} bits ({len(secret_bits)//8} bytes)"
    )
    print(f"[*] Edge pixels: {edge_pixels}, Non-edge: {non_edge_pixels}")

    idx_bit = 0
    bits_embedded = 0

    for j in range(h):
        for i in range(w):
            if idx_bit >= len(secret_bits):
                break

            pixel = stego[j, i].copy()
            is_edge = 1 if edge_map[j, i] > 0 else 0
            M = min(x, 8) if is_edge else min(y, 8)

            # Nhúng data vào từng channel
            for ch in range(c):
                for b in range(M):
                    if idx_bit >= len(secret_bits):
                        break

                    # Clear bit thứ b và set giá trị mới
                    mask = 0xFF ^ (1 << b)
                    pixel[ch] = (pixel[ch] & mask) | (int(secret_bits[idx_bit]) << b)
                    idx_bit += 1
                    bits_embedded += 1

            stego[j, i] = pixel

        if idx_bit >= len(secret_bits):
            break

    return stego, bits_embedded


# ---------------- Extract Message ----------------
def extract_message_from_stego(stego_img, edge_map, message_length_bits, x=3, y=1):
    """
    Trích xuất 'message_length_bits' bit từ ảnh stego
    (tính từ LSB theo đúng thứ tự nhúng).

    Args:
        stego_img: ảnh stego (BGR, uint8) hoặc đường dẫn
        edge_map: edge map (grayscale, phải giống lúc embed) hoặc đường dẫn
        message_length_bits: số bit cần đọc (header hoặc header+msg)
        x: số bit đã nhúng mỗi channel ở vùng edge
        y: số bit đã nhúng mỗi channel ở vùng non-edge

    Returns:
        message_bits: list bit đọc được (không xử lý header ở đây).
    """
    # Đọc ảnh nếu là đường dẫn
    if isinstance(stego_img, str):
        stego_img = cv2.imread(stego_img)
        if stego_img is None:
            raise ValueError("Không thể đọc ảnh stego")

    if isinstance(edge_map, str):
        edge_map = cv2.imread(edge_map, cv2.IMREAD_GRAYSCALE)
        if edge_map is None:
            raise ValueError("Không thể đọc edge map")

    if stego_img is None:
        raise ValueError("stego_img is None")
    if edge_map is None:
        raise ValueError("edge_map is None")

    # Validate
    if len(stego_img.shape) != 3 or stego_img.shape[2] != 3:
        raise ValueError("Ảnh phải có 3 channels (BGR)")

    if edge_map.shape[:2] != stego_img.shape[:2]:
        raise ValueError("Edge map phải có cùng kích thước với ảnh")

    if not (1 <= x <= 8 and 1 <= y <= 8):
        raise ValueError(f"x và y phải trong khoảng 1-8, nhận được x={x}, y={y}")

    h, w, c = stego_img.shape
    message_bits = []

    for j in range(h):
        for i in range(w):
            if len(message_bits) >= message_length_bits:
                break

            pixel = stego_img[j, i]
            is_edge = 1 if edge_map[j, i] > 0 else 0
            M = min(x, 8) if is_edge else min(y, 8)

            # Trích xuất bit từ từng channel
            for ch in range(c):
                ch_val = int(pixel[ch])

                # Đọc M bit ĐẦU TIÊN (LSB 0, 1, 2, ... M-1)
                for b in range(M):
                    if len(message_bits) >= message_length_bits:
                        break

                    bit = (ch_val >> b) & 1
                    message_bits.append(bit)

        if len(message_bits) >= message_length_bits:
            break

    return message_bits


# ========== WRAPPER FUNCTIONS cho PyQt6 ==========

def embed_message(cover_img_or_path, edge_map, secret_message, output_path, 
                  algorithm, x=3, y=1):
    """
    Hàm wrapper để nhúng TEXT message vào ảnh (dùng trong EmbedWindow).

    Format bit nhúng:
        [ALGORITHM_ID (8 bits - cố định)] [HEADER_LENGTH (32 bits)] [MESSAGE_DATA bits]

    Args:
        cover_img_or_path: đường dẫn ảnh gốc hoặc numpy array
        edge_map: edge map (numpy array, do EmbedWindow tạo sẵn)
        secret_message: string cần nhúng (tiếng Anh)
        output_path: đường dẫn lưu ảnh stego
        algorithm: tên thuật toán ("sobel"/"canny"/"hybrid")
        x, y: số bit nhúng ở vùng edge / non-edge

    Returns:
        bits_embedded: tổng số bit đã nhúng (8 + header + message)
    """
    print("\n" + "="*60)
    print("BẮT ĐẦU NHÚNG MESSAGE")
    print("="*60)
    
    # 0. Đọc ảnh nếu là path
    if isinstance(cover_img_or_path, str):
        img_cover = cv2.imread(cover_img_or_path)
        if img_cover is None:
            raise ValueError(f"Không thể đọc ảnh: {cover_img_or_path}")
    else:
        img_cover = cover_img_or_path.copy()
    
    print(f"[1] Ảnh cover shape: {img_cover.shape}")
    print(f"    Pixel (0,0) ban đầu: {img_cover[0,0]}")
    
    # 1. Chuyển algorithm thành 8 bits (LSB first)
    algo_id = algorithm_to_id(algorithm)
    algo_bits_msb = int_to_bits(algo_id, 8)  # MSB first: [0,0,0,0,0,0,0,1] cho ID=1
    algo_bits_lsb = algo_bits_msb[::-1]       # LSB first: [1,0,0,0,0,0,0,0]
    
    print(f"[2] Algorithm: {algorithm} (ID={algo_id})")
    print(f"    Bits MSB first: {algo_bits_msb}")
    print(f"    Bits LSB first: {algo_bits_lsb}")
    
    # 2. Text -> bits
    message_bits = text_to_bits(secret_message)
    message_length_bits = len(message_bits)
    
    print(f"[3] Message: '{secret_message[:20]}...'")
    print(f"    Length: {message_length_bits} bits ({message_length_bits//8} bytes)")

    # 3. Header 32 bit chứa độ dài message
    header_bits = int_to_bits(message_length_bits, 32)
    print(f"[4] Header (32 bits): {header_bits[:10]}... (message length)")

    # 4. Ghép: header + message (chưa có algo_bits, vì sẽ nhúng riêng)
    secret_bits_mlsb = header_bits + message_bits

    # 5. Nhúng 8 bits algorithm vào vị trí cố định
    print(f"[5] Nhúng 8 bits algorithm vào pixel (0,0)...")
    img_cover = embed_fixed_8bits(img_cover, algo_bits_lsb)
    
    # 6. Nhúng phần còn lại bằng MLSB với edge map
    print(f"[6] Nhúng {len(secret_bits_mlsb)} bits (header+message) bằng MLSB...")
    stego_img, bits_embedded = embed_mlsb(
        img_cover,  # ảnh đã có algorithm_id
        edge_map,
        secret_bits_mlsb,  # header + message
        x,
        y,
    )

    # 7. Lưu ảnh
    cv2.imwrite(output_path, stego_img)
    print(f"[7] ✅ Đã lưu ảnh stego tại: {output_path}")
    print(f"    Tổng bits nhúng: {8 + bits_embedded} (8 algo + {bits_embedded} data)")
    print("="*60 + "\n")

    return 8 + bits_embedded  # Trả về tổng số bits (bao gồm 8 bits algorithm)


def extract_message(stego_img, edge_map, expected_algorithm, x=3, y=1):
    """
    Hàm wrapper để trích xuất TEXT message từ ảnh stego
    (dùng trong ExtractWindow).

    Args:
        stego_img: đường dẫn ảnh stego hoặc numpy array
        edge_map: đường dẫn edge map hoặc numpy array
        expected_algorithm: thuật toán người dùng chọn ("sobel"/"canny"/"hybrid")
        x, y: số bit đã nhúng ở vùng edge / non-edge

    Returns:
        message: string đã giải mã
        
    Raises:
        ValueError: nếu thuật toán không khớp
    """
    print("\n" + "="*60)
    print("BẮT ĐẦU TRÍCH XUẤT MESSAGE")
    print("="*60)
    
    # 1. Đọc ảnh nếu là path
    if isinstance(stego_img, str):
        print(f"[1] Đọc ảnh stego từ: {stego_img}")
        stego_img = cv2.imread(stego_img)
        if stego_img is None:
            raise ValueError("Không thể đọc ảnh stego")
    else:
        print(f"[1] Sử dụng numpy array, shape: {stego_img.shape}")

    if isinstance(edge_map, str):
        print(f"[2] Đọc edge map từ: {edge_map}")
        edge_map = cv2.imread(edge_map, cv2.IMREAD_GRAYSCALE)
        if edge_map is None:
            raise ValueError("Không thể đọc edge map")
    else:
        print(f"[2] Sử dụng edge map array, shape: {edge_map.shape}")

    # 2. Trích xuất 8 bits algorithm từ vị trí cố định
    print(f"[3] Trích xuất 8 bits algorithm từ pixel (0,0)...")
    
    algo_bits_lsb = extract_fixed_8bits(stego_img)
    algo_bits_msb = algo_bits_lsb[::-1]  # Chuyển về MSB first để decode
    
    print(f"    Bits LSB first: {algo_bits_lsb}")
    print(f"    Bits MSB first: {algo_bits_msb}")
    
    algo_id = bits_to_int(algo_bits_msb)
    print(f"    Algorithm ID: {algo_id}")
    
    try:
        embedded_algorithm = id_to_algorithm(algo_id)
        print(f"    Algorithm name: {embedded_algorithm}")
    except ValueError as e:
        print(f"    ❌ LỖI: {e}")
        raise ValueError(
            f"❌ Không thể xác định thuật toán từ ảnh!\n"
            f"Algorithm ID đọc được: {algo_id} (không hợp lệ)\n"
            f"Ảnh này có thể:\n"
            f"  - Không phải ảnh stego của chương trình này\n"
            f"  - Đã bị chỉnh sửa/nén sau khi nhúng"
        )
    
    print(f"[4] So sánh thuật toán:")
    print(f"    - Nhúng trong ảnh: {embedded_algorithm.upper()}")
    print(f"    - Người dùng chọn: {expected_algorithm.upper()}")
    
    # 3. So sánh với thuật toán người dùng chọn
    if embedded_algorithm.lower() != expected_algorithm.lower():
        print(f"    ❌ KHÔNG KHỚP!")
        raise ValueError(
            f"❌ SAI THUẬT TOÁN!\n\n"
            f"Thuật toán đã dùng khi nhúng: {embedded_algorithm.upper()}\n"
            f"Thuật toán bạn đang chọn: {expected_algorithm.upper()}\n\n"
            f"Vui lòng chọn đúng thuật toán để trích xuất!"
        )
    
    print(f"    ✅ KHỚP! Tiếp tục trích xuất...")
    
    # 4. Đọc header 32 bit từ MLSB
    print(f"[5] Đọc 32 bits header...")
    header_bits = extract_message_from_stego(stego_img, edge_map, 32, x=x, y=y)
    print(f"    Header bits (10 đầu): {header_bits[:10]}...")
    
    message_length_bits = bits_to_int(header_bits)
    print(f"    Message length: {message_length_bits} bits ({message_length_bits//8} bytes)")

    if message_length_bits <= 0 or message_length_bits > 1_000_000:
        raise ValueError(
            f"❌ Độ dài message không hợp lệ: {message_length_bits} bits"
        )

    # 5. Đọc lại: 32 (header) + message_length_bits
    print(f"[6] Đọc tổng {32 + message_length_bits} bits (header + message)...")
    all_bits = extract_message_from_stego(
        stego_img,
        edge_map,
        32 + message_length_bits,
        x=x,
        y=y,
    )

    # 6. Bỏ 32 bit header, lấy đúng số bit message
    message_bits = all_bits[32 : 32 + message_length_bits]
    print(f"[7] Message bits (10 đầu): {message_bits[:10]}...")

    # 7. Chuyển về text
    message = bits_to_text(message_bits)
    print(f"[8] ✅ Đã trích xuất message: {len(message)} ký tự")
    print(f"    Message: '{message[:50]}...'")
    print("="*60 + "\n")

    return message


# ----------------- Metrics -----------------
def compute_metrics(original_path, stego_path, bits_embedded):
    """
    Trả về (psnr, bits_embedded, bpp)
    bpp = bits_embedded / (H * W)
    """
    img1 = cv2.imread(original_path).astype(np.float64)
    img2 = cv2.imread(stego_path).astype(np.float64)

    if img1 is None or img2 is None:
        raise ValueError("Không thể đọc ảnh")

    if img1.shape != img2.shape:
        raise ValueError("Original và stego phải có cùng kích thước")

    mse = np.mean((img1 - img2) ** 2)
    psnr = float("inf") if mse == 0 else 10 * math.log10((255.0**2) / mse)

    H, W, _ = img1.shape
    bpp = bits_embedded / (H * W)

    return psnr, bits_embedded, bpp