# core/stego_embed.py
import cv2
import numpy as np
import os
import math

# ----------------- Helpers -----------------
def read_file_binary(file_path):
    with open(file_path, "rb") as f:
        return f.read()

def to_binary(data, bit_length=None):
    """Chuyển bytes hoặc int sang chuỗi '0101...'"""
    if isinstance(data, (bytes, bytearray)):
        return ''.join(format(b, '08b') for b in data)
    if isinstance(data, int):
        if bit_length:
            return format(data, f'0{bit_length}b')
        return format(data, '032b')
    raise TypeError("to_binary: data phải là bytes hoặc int")

def _set_bits_in_str(bin_str, start_idx, bits):
    """Thay thế phần bin_str[start_idx:start_idx+len(bits)] bằng bits.
       start_idx có thể âm (từ cuối)."""
    # normalize indices for python slicing
    n = len(bin_str)
    if start_idx < 0:
        start_idx = n + start_idx
    end_idx = start_idx + len(bits)
    return bin_str[:start_idx] + bits + bin_str[end_idx:]

# ----------------- Embed -----------------
def embed_message(cover_img_or_path, edge_map, secret_file_path, output_path):
    """
    Nhúng file (bất kỳ) vào ảnh theo quy tắc:
      - Pixel biên: flag R[-1]=1, R[-4:-1] <- 3 bits, G[-3:] <-3, B[-3:] <-3 (9 bits)
      - Pixel nền: flag R[-1]=0, R[-2] <-1 bit, G[-1] <-1, B[-1] <-1 (3 bits)
    Header: [8bit ext_len][ext_bytes][32bit file_size][file_data_bits]
    Trả về: số bit đã nhúng (int)
    """
    # load cover image
    cover_img = cv2.imread(cover_img_or_path) if isinstance(cover_img_or_path, str) else cover_img_or_path
    if cover_img is None:
        raise ValueError("Không đọc được ảnh bìa")

    # load edge_map
    if isinstance(edge_map, str):
        edge_map = cv2.imread(edge_map, cv2.IMREAD_GRAYSCALE)
    if edge_map is None or edge_map.shape[:2] != cover_img.shape[:2]:
        raise ValueError("edge_map không hợp lệ hoặc không khớp kích thước ảnh")

    # read file
    file_data = read_file_binary(secret_file_path)
    file_size = len(file_data)
    file_ext = os.path.splitext(secret_file_path)[1].strip() or ".bin"
    # sanitize ext
    file_ext = ''.join(ch for ch in file_ext if ch.isalnum() or ch in ['.', '_'])
    if not file_ext.startswith('.'):
        file_ext = '.' + file_ext
    ext_bytes = file_ext.encode('utf-8')
    ext_len = len(ext_bytes)

    # build payload bits
    payload_bits = (
        to_binary(ext_len, 8) +
        ''.join(format(b, '08b') for b in ext_bytes) +   # ✅ encode từng byte đúng chuẩn
        to_binary(file_size, 32) +
        to_binary(file_data)
    )
    total_bits = len(payload_bits)

    # compute capacity from edge_map (9 bits per edge pixel, 3 bits per non-edge)
    rows, cols = edge_map.shape[:2]
    capacity_bits = 0
    for i in range(rows):
        for j in range(cols):
            capacity_bits += 9 if edge_map[i, j] > 0 else 3
    if total_bits > capacity_bits:
        raise ValueError(f"Ảnh không đủ dung lượng: cần {total_bits} bits, capacity {capacity_bits} bits")

    stego = cover_img.copy()
    bit_index = 0

    # iterate pixels row-major
    for i in range(rows):
        for j in range(cols):
            if bit_index >= total_bits:
                break
            pixel = list(stego[i, j])  # B,G,R in OpenCV order
            # note: pixel[2] is R, pixel[1] is G, pixel[0] is B
            is_edge = bool(edge_map[i, j] > 0)

            # ----- set flag in R LSB -----
            r_bin = format(pixel[2], '08b')
            flag_bit = '1' if is_edge else '0'
            r_bin = r_bin[:-1] + flag_bit  # keep other bits, set LSB = flag
            pixel[2] = int(r_bin, 2)

            if is_edge:
                # EDGE: embed 3 bits into R[-4:-1], 3 into G[-3:], 3 into B[-3:]
                # R: replace bits index -4..-2 (3 bits before final LSB)
                # get 3 bits for R
                bits_r = payload_bits[bit_index: bit_index + 3].ljust(3, '0')
                bit_index += min(3, total_bits - (bit_index - 3))  # adjust below not necessary; will cap after
                # careful: set R[-4:-1] (positions -4,-3,-2)
                r_bin = format(pixel[2], '08b')
                r_bin = r_bin[:-4] + bits_r + r_bin[-1]  # keep final LSB (flag)
                pixel[2] = int(r_bin, 2)

                # G: 3 bits into G[-3:]
                bits_g = payload_bits[bit_index: bit_index + 3].ljust(3, '0')
                bit_index += min(3, total_bits - bit_index + 3)  # not strictly needed
                g_bin = format(pixel[1], '08b')
                g_bin = g_bin[:-3] + bits_g
                pixel[1] = int(g_bin, 2)

                # B: 3 bits into B[-3:]
                bits_b = payload_bits[bit_index: bit_index + 3].ljust(3, '0')
                bit_index += min(3, total_bits - bit_index + 3)
                b_bin = format(pixel[0], '08b')
                b_bin = b_bin[:-3] + bits_b
                pixel[0] = int(b_bin, 2)

                # Note: above bit_index increments are cautious but to avoid confusion,
                # better to increment strictly by actual inserted bits (below we'll correct).
                # We'll instead manage insertion properly by slicing the payload.
            else:
                # NON-EDGE: embed 1 bit into R[-2], 1 bit into G[-1], 1 bit into B[-1]
                # R[-2]:
                bits_r = payload_bits[bit_index: bit_index + 1].ljust(1, '0')
                bit_index += min(1, total_bits - (bit_index - 1))
                r_bin = format(pixel[2], '08b')
                # place at index -2
                r_bin = r_bin[:-2] + bits_r + r_bin[-1]
                pixel[2] = int(r_bin, 2)

                # G[-1]
                bits_g = payload_bits[bit_index: bit_index + 1].ljust(1, '0')
                bit_index += min(1, total_bits - bit_index + 1)
                g_bin = format(pixel[1], '08b')
                g_bin = g_bin[:-1] + bits_g
                pixel[1] = int(g_bin, 2)

                # B[-1]
                bits_b = payload_bits[bit_index: bit_index + 1].ljust(1, '0')
                bit_index += min(1, total_bits - bit_index + 1)
                b_bin = format(pixel[0], '08b')
                b_bin = b_bin[:-1] + bits_b
                pixel[0] = int(b_bin, 2)

            # BUT the above increments are messy. To be exact, we must consume payload bits in exact order:
            # We'll rewrite this loop more deterministically below to avoid off-by-one. (See final implementation.)

            stego[i, j] = tuple(pixel)

        if bit_index >= total_bits:
            break

    # The above attempted incremental approach risks off-by-one due to the min(...) mix.
    # Simpler: implement a clean loop below (rebuild with proper consumption).
    # We'll implement a precise version now (override previous result).

    # Precise clean implementation:
    stego = cover_img.copy()
    bit_index = 0
    rows, cols, _ = stego.shape

    for i in range(rows):
        for j in range(cols):
            if bit_index >= total_bits:
                break
            pixel = list(stego[i, j])
            is_edge = bool(edge_map[i, j] > 0)

            # Set flag in R LSB
            r_bin = format(pixel[2], '08b')
            flag_bit = '1' if is_edge else '0'
            r_bin = r_bin[:-1] + flag_bit
            pixel[2] = int(r_bin, 2)

            if is_edge:
                # R: next 3 bits go into positions -4..-2
                bits_r = payload_bits[bit_index: bit_index + 3]
                bits_r = bits_r.ljust(3, '0')
                bit_index += min(3, total_bits - (bit_index))
                r_bin = format(pixel[2], '08b')
                r_bin = r_bin[:-4] + bits_r + r_bin[-1]
                pixel[2] = int(r_bin, 2)

                # G: 3 bits to G[-3:]
                bits_g = payload_bits[bit_index: bit_index + 3]
                bits_g = bits_g.ljust(3, '0')
                bit_index += min(3, total_bits - (bit_index))
                g_bin = format(pixel[1], '08b')
                g_bin = g_bin[:-3] + bits_g
                pixel[1] = int(g_bin, 2)

                # B: 3 bits to B[-3:]
                bits_b = payload_bits[bit_index: bit_index + 3]
                bits_b = bits_b.ljust(3, '0')
                bit_index += min(3, total_bits - (bit_index))
                b_bin = format(pixel[0], '08b')
                b_bin = b_bin[:-3] + bits_b
                pixel[0] = int(b_bin, 2)

            else:
                # non-edge: R[-2], G[-1], B[-1]
                bits_r = payload_bits[bit_index: bit_index + 1]
                bits_r = bits_r.ljust(1, '0')
                bit_index += min(1, total_bits - (bit_index))
                r_bin = format(pixel[2], '08b')
                r_bin = r_bin[:-2] + bits_r + r_bin[-1]
                pixel[2] = int(r_bin, 2)

                bits_g = payload_bits[bit_index: bit_index + 1]
                bits_g = bits_g.ljust(1, '0')
                bit_index += min(1, total_bits - (bit_index))
                g_bin = format(pixel[1], '08b')
                g_bin = g_bin[:-1] + bits_g
                pixel[1] = int(g_bin, 2)

                bits_b = payload_bits[bit_index: bit_index + 1]
                bits_b = bits_b.ljust(1, '0')
                bit_index += min(1, total_bits - (bit_index))
                b_bin = format(pixel[0], '08b')
                b_bin = b_bin[:-1] + bits_b
                pixel[0] = int(b_bin, 2)

            stego[i, j] = tuple(pixel)
        if bit_index >= total_bits:
            break

    ok = cv2.imwrite(output_path, stego)
    if not ok:
        raise IOError("Không lưu được ảnh stego")
    return bit_index

# ----------------- Extract -----------------
def extract_message(stego_img_or_path, edge_map, output_dir, debug=False):
    """
    Trích xuất file đã nhúng. Trả về đường dẫn file đã giải mã.
    An toàn hơn: loại bỏ ký tự điều khiển/không hợp lệ trong extension,
    xử lý ext_len = 0, ext_bits ngắn, và đảm bảo tên file hợp lệ trên Windows.
    """
    stego = cv2.imread(stego_img_or_path) if isinstance(stego_img_or_path, str) else stego_img_or_path
    if stego is None:
        raise ValueError("Không đọc được ảnh stego")

    if isinstance(edge_map, str):
        edge_map = cv2.imread(edge_map, cv2.IMREAD_GRAYSCALE)
    if edge_map is None or edge_map.shape[:2] != stego.shape[:2]:
        raise ValueError("edge_map không hợp lệ hoặc không khớp kích thước ảnh")

    rows, cols, _ = stego.shape

    def read_n_bits(n):
        """Đọc n bit tiếp theo theo thứ tự pixel, sử dụng flag bit ở R[-1]"""
        bits_collected = ""
        for i in range(rows):
            for j in range(cols):
                if len(bits_collected) >= n:
                    return bits_collected[:n]
                pixel = stego[i, j]
                # pixel is B,G,R
                r_bin = format(int(pixel[2]), '08b')
                flag = r_bin[-1]
                if flag == '1':
                    # edge: R[-4:-1], G[-3:], B[-3:]
                    r_bits = r_bin[-4:-1]  # 3 bits
                    g_bits = format(int(pixel[1]), '08b')[-3:]
                    b_bits = format(int(pixel[0]), '08b')[-3:]
                    bits_collected += r_bits + g_bits + b_bits
                else:
                    # non-edge: R[-2], G[-1], B[-1]
                    r_bit = r_bin[-2]
                    g_bit = format(int(pixel[1]), '08b')[-1]
                    b_bit = format(int(pixel[0]), '08b')[-1]
                    bits_collected += r_bit + g_bit + b_bit
                if len(bits_collected) >= n:
                    return bits_collected[:n]
        return bits_collected[:n]

    # 1) 8 bits ext_len
    ext_len_bits = read_n_bits(8)
    if len(ext_len_bits) < 8:
        raise ValueError("Không đọc đủ 8 bit ext_len")
    ext_len = int(ext_len_bits, 2)

    # safety: cap ext_len (chống lưu giá trị quá lớn do lỗi)
    if ext_len < 0 or ext_len > 64:
        if debug:
            print(f"[WARN] ext_len bất thường ({ext_len}) -> dùng 0")
        ext_len = 0

    # 2) ext_len * 8 bits -> ext bytes
    ext_bits = read_n_bits(ext_len * 8) if ext_len > 0 else ""
    if len(ext_bits) < ext_len * 8 and debug:
        print("[WARN] Không đọc đủ bits cho extension (đã đọc {}/{} bits)".format(len(ext_bits), ext_len*8))

    ext_bytes = bytearray(int(ext_bits[i:i+8], 2) for i in range(0, len(ext_bits), 8)) if ext_bits else bytearray()
    try:
        ext_raw = ext_bytes.decode('utf-8', errors='ignore')
    except Exception:
        ext_raw = ''.join(chr(b) for b in ext_bytes if 32 <= b < 127)

    if debug:
        print("[DEBUG] ext_raw (raw):", repr(ext_raw))

    # --- Sanitize extension: loại bỏ control chars và chỉ giữ ký tự hợp lệ ---
    # cho phép: letters, digits, dot, underscore
    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"
    ext_clean = ''.join(ch for ch in ext_raw if ch in valid_chars)

    # trim và bảo đảm có dot
    ext_clean = ext_clean.strip()
    if ext_clean == "":
        ext = ".bin"
    else:
        if not ext_clean.startswith('.'):
            ext = '.' + ext_clean
        else:
            ext = ext_clean

    # hạn chế độ dài ext (an toàn), và nếu không có chữ cái -> .bin
    if len(ext) < 3 or len(ext) > 16 or not any(ch.isalpha() for ch in ext):
        if debug:
            print("[WARN] ext không hợp lệ sau làm sạch -> dùng .bin (ext_raw={!r} -> ext_clean={!r})".format(ext_raw, ext_clean))
        ext = '.bin'

    # 3) 32 bits file_size
    size_bits = read_n_bits(32)
    if len(size_bits) < 32:
        raise ValueError("Không đọc đủ 32 bit cho file size")
    file_size = int(size_bits, 2)

    if debug:
        print(f"[DEBUG] ext={ext}, file_size={file_size}")

    # 4) file_size * 8 bits data
    data_bits = read_n_bits(file_size * 8)
    if len(data_bits) < file_size * 8 and debug:
        print(f"[WARN] dữ liệu file bị thiếu ({len(data_bits)} / {file_size*8} bits)")

    # convert to bytes
    data_bytes = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = data_bits[i:i+8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        data_bytes.append(int(byte, 2))

    # đảm bảo output_dir tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # tạo tên file an toàn: "extracted_file" + ext, nhưng escape/normalize đường dẫn
    filename = "extracted_file" + ext
    # loại bỏ ký tự không hợp lệ trong filename (phòng trường hợp ext vẫn còn rác)
    safe_name_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    # giữ dot của ext, nhưng cho filename phần trước chỉ chứa safe chars
    base_name = ''.join(ch for ch in "extracted_file" if ch in safe_name_chars)
    filename = base_name + ext
    out_path = os.path.normpath(os.path.join(output_dir, filename))

    # cuối cùng ghi file
    with open(out_path, "wb") as f:
        f.write(data_bytes)

    return out_path

# ----------------- Metrics -----------------
def compute_metrics(original_path, stego_path, bits_embedded):
    """
    Trả về (psnr, bits_embedded, bpp)
    bpp = bits_embedded / (H * W)
    """
    img1 = cv2.imread(original_path).astype(np.float64)
    img2 = cv2.imread(stego_path).astype(np.float64)
    if img1.shape != img2.shape:
        raise ValueError("Original và stego phải có cùng kích thước")
    mse = np.mean((img1 - img2) ** 2)
    psnr = float('inf') if mse == 0 else 10 * math.log10((255.0 ** 2) / mse)
    H, W, _ = img1.shape
    bpp = bits_embedded / (H * W)
    return psnr, bits_embedded, bpp
