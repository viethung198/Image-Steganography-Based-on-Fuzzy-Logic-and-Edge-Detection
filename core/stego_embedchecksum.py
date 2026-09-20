import cv2
import numpy as np
import math
import hashlib

# ---------------- Bit Conversion ----------------
def text_to_bits(text: str):
    """
    Chuyển text (ASCII / English) thành list các bit.
    Mỗi ký tự -> 8 bit (ord/chr) giống code console của bạn.
    """
    return [int(b) for c in text for b in format(ord(c), "08b")]


def int_to_bits(n: int, bit_length: int = 32):
    return [int(b) for b in format(n, f"0{bit_length}b")]


def bits_to_int(bits):
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


# ---------------- Edge Map Checksum ----------------
def compute_edge_checksum(edge_map):
    """
    Tính checksum (hash) của edge map để verify khi extract.
    Dùng SHA256 và lấy 32 bit đầu.
    
    Args:
        edge_map: numpy array (grayscale)
        
    Returns:
        checksum: int 32-bit
    """
    edge_bytes = edge_map.tobytes()
    hash_obj = hashlib.sha256(edge_bytes)
    checksum = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
    return checksum


# ---------------- LSB Embedding ----------------
def embed_mlsb(img_cover, edge_map, secret_bits, x=3, y=1):
    """
    Nhúng tin nhắn vào ảnh sử dụng MLSB với edge detection.

    KHÔNG tự tạo header trong hàm này.
    secret_bits đã bao gồm header (32 bit edge checksum + 32 bit length) + message.

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

def embed_message(cover_img_or_path, edge_map, secret_message, output_path, x=3, y=1):
    """
    Hàm wrapper để nhúng TEXT message vào ảnh (dùng trong EmbedWindow).

    Format bit nhúng:
        [EDGE_CHECKSUM (32 bits)] [MESSAGE_LENGTH (32 bits)] [MESSAGE_DATA bits]

    Args:
        cover_img_or_path: đường dẫn ảnh gốc hoặc numpy array
        edge_map: edge map (numpy array, do EmbedWindow tạo sẵn)
        secret_message: string cần nhúng (tiếng Anh)
        output_path: đường dẫn lưu ảnh stego
        x, y: số bit nhúng ở vùng edge / non-edge

    Returns:
        bits_embedded: tổng số bit đã nhúng (edge checksum + length + message)
    """
    # 1. Tính edge checksum
    edge_checksum = compute_edge_checksum(edge_map)
    checksum_bits = int_to_bits(edge_checksum, 32)
    print(f"[*] Edge map checksum: {edge_checksum:08x}")
    
    # 2. Text -> bits
    message_bits = text_to_bits(secret_message)
    message_length_bits = len(message_bits)

    # 3. Header 32 bit chứa độ dài message (tính theo bit)
    length_bits = int_to_bits(message_length_bits, 32)

    # 4. Ghép: checksum + length + data
    secret_bits_full = checksum_bits + length_bits + message_bits

    # 5. Nhúng
    stego_img, bits_embedded = embed_mlsb(
        cover_img_or_path,
        edge_map,
        secret_bits_full,
        x,
        y,
    )

    # 6. Lưu ảnh
    cv2.imwrite(output_path, stego_img)
    print(f"[✓] Đã lưu ảnh stego tại: {output_path}")

    return bits_embedded


def extract_message(stego_img, edge_map, x=3, y=1):
    """
    Hàm wrapper để trích xuất TEXT message từ ảnh stego
    (dùng trong ExtractWindow).

    Args:
        stego_img: đường dẫn ảnh stego hoặc numpy array
        edge_map: đường dẫn edge map hoặc numpy array
        x, y: số bit đã nhúng ở vùng edge / non-edge

    Returns:
        message: string đã giải mã
    """
    # 1. Đọc edge checksum (32 bit đầu tiên)
    checksum_bits = extract_message_from_stego(stego_img, edge_map, 32, x=x, y=y)
    stored_checksum = bits_to_int(checksum_bits)
    
    # 2. Verify edge map
    current_checksum = compute_edge_checksum(edge_map)
    
    print(f"[*] Stored edge checksum: {stored_checksum:08x}")
    print(f"[*] Current edge checksum: {current_checksum:08x}")
    
    if stored_checksum != current_checksum:
        raise ValueError(
            f"❌ Edge map KHÔNG KHỚP!\n"
            f"   - Checksum lưu trong stego: {stored_checksum:08x}\n"
            f"   - Checksum edge map hiện tại: {current_checksum:08x}\n"
            f"\n"
            f"🔧 Nguyên nhân có thể:\n"
            f"   1. Bạn dùng SAI thuật toán edge detection (Sobel ≠ Canny ≠ Fuzzy)\n"
            f"   2. Bạn dùng SAI threshold/parameters cho edge detection\n"
            f"   3. Edge map bị thay đổi sau khi nhúng\n"
            f"\n"
            f"💡 Giải pháp: Phải dùng ĐÚNG edge map y hệt lúc nhúng!"
        )
    
    print("[✓] Edge map checksum khớp!")
    
    # 3. Đọc message length (32 bit tiếp theo)
    length_bits = extract_message_from_stego(stego_img, edge_map, 64, x=x, y=y)[32:64]
    message_length_bits = bits_to_int(length_bits)

    print(
        f"[*] Message length from header: "
        f"{message_length_bits} bits ({message_length_bits//8} bytes)"
    )

    if message_length_bits <= 0 or message_length_bits > 10_000_000:
        raise ValueError(
            f"Độ dài message không hợp lệ: {message_length_bits} bits\n"
            f"Header có thể đã bị lỗi hoặc tham số x,y không khớp lúc nhúng."
        )

    # 4. Đọc toàn bộ: checksum(32) + length(32) + message
    total_bits = 64 + message_length_bits
    all_bits = extract_message_from_stego(
        stego_img,
        edge_map,
        total_bits,
        x=x,
        y=y,
    )

    # 5. Bỏ 64 bit header (checksum + length), lấy đúng số bit message
    message_bits = all_bits[64 : 64 + message_length_bits]

    # 6. Chuyển về text
    message = bits_to_text(message_bits)
    print(f"[✓] Đã trích xuất message: {len(message)} ký tự")

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