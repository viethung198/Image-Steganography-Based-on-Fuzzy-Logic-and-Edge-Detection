import os
import cv2
from main_edge_detection import run_edge_detection
from core.stego_embed import read_file_content, embed_message, extract_message, compute_metrics
import tkinter as tk
from tkinter import filedialog

# =========================================
# Hàm chọn file với hộp thoại nổi (đè lên VSCode)
# =========================================
def choose_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()                      # Ẩn cửa sổ Tk gốc
    root.attributes('-topmost', True)    # Luôn nổi trên cùng
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=root)
    root.destroy()                       # Giải phóng sau khi chọn
    return file_path


def main():
    print("===== HỆ THỐNG PHÁT HIỆN CẠNH & ẨN MÃ =====")

    # Gọi phát hiện cạnh (trả về đường dẫn ảnh đã chọn và edge_map ndarray)
    result = run_edge_detection(return_result=True)

    if result is None:
        print("❌ Không có ảnh nào được phát hiện cạnh, thoát.")
        return

    img_path, edge_map = result  # img_path là đường dẫn (str), edge_map là ndarray (grayscale)
    print(f"[DEBUG] Ảnh: {img_path}, edge_map shape: {edge_map.shape if edge_map is not None else 'None'}")

    # đọc cover image
    cover_img = cv2.imread(img_path)
    if cover_img is None:
        print("❌ Không đọc được ảnh gốc.")
        return

    choice = input("\nBạn có muốn giấu tin vào ảnh này không? (y/n): ").strip().lower()
    if choice == 'y':
        # chọn file bằng dialog (đè lên VSCode)
        print("🗂️ Chọn file cần giấu (.txt hoặc .docx)...")
        file_path = choose_file("Chọn file cần giấu (.txt hoặc .docx)",
                                [("Text files", "*.txt"), ("Word files", "*.docx")])
        if not file_path:
            print("❌ Bạn chưa chọn file, thoát.")
            return

        secret = read_file_content(file_path)
        if not secret.strip():
            print("⚠️ File không có nội dung để giấu, thoát.")
            return

        output_dir = "output_stego"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "stego_image.png")

        print("[+] Đang giấu nội dung file vào ảnh...")
        try:
            used_pixels, used_bits = embed_message(img_path, edge_map, secret, output_path)
        except Exception as e:
            print("❌ Lỗi khi giấu tin:", e)
            return

        print(f"✅ Ảnh giấu tin đã lưu: {output_path}")
        # tính metrics (chú ý compute_metrics yêu cầu bits_embedded)
        try:
            psnr, payload, ratio = compute_metrics(img_path, output_path, used_bits)
            print(f"📊 PSNR: {psnr:.2f} dB")
            print(f"📦 Payload: {payload} bits")
            total_pixels = cover_img.shape[0] * cover_img.shape[1]
            print(f"🟩 Đã nhúng: {used_pixels} pixel (≈ {used_pixels/total_pixels*100:.4f}% tổng pixel)")
            print(f"📈 Ratio (payload/total_bits): {ratio:.6f}")
        except Exception as e:
            print("⚠️ Lỗi khi tính metrics:", e)

    else:
        decode_choice = input("Bạn có muốn giải mã ảnh stego không? (y/n): ").strip().lower()
        if decode_choice == 'y':
            print("🖼️ Chọn ảnh stego cần giải mã (.png hoặc .jpg)...")
            stego_path = choose_file("Chọn ảnh stego cần giải mã (.png hoặc .jpg)",
                                     [("Image files", "*.png *.jpg *.jpeg")])
            if not stego_path:
                print("❌ Bạn chưa chọn ảnh, thoát.")
                return

            # hỏi cách lấy edge_map cho ảnh stego
            auto = input("Bạn có muốn dùng phương pháp phát hiện cạnh (Canny) tự động để tạo edge_map cho ảnh stego không? (y/n): ").strip().lower()
            if auto == 'y':
                # tính edge map nhanh bằng Canny
                img_gray = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    print("❌ Không đọc được ảnh stego.")
                    return
                edge_map_decode = cv2.Canny(img_gray, 100, 200)
                print("[+] Đã tạo edge_map bằng Canny cho ảnh stego.")
            else:
                print("🗂️ Chọn file edge_map tương ứng (grayscale) ...")
                edge_file = choose_file("Chọn edge map tương ứng (grayscale)",
                                        [("Image files", "*.png *.jpg *.jpeg")])
                if not edge_file:
                    print("❌ Bạn chưa chọn edge_map, thoát.")
                    return
                edge_map_decode = cv2.imread(edge_file, cv2.IMREAD_GRAYSCALE)

            try:
                message = extract_message(stego_path, edge_map_decode)
                if message:
                    print("\n🔓 Thông điệp giải mã được:")
                    print("-------------------------------------------------")
                    print(message)
                    print("-------------------------------------------------")
                    save = input("Bạn có muốn lưu thông điệp này ra file .txt không? (y/n): ").strip().lower()
                    if save == 'y':
                        with open("decoded_message.txt", "w", encoding="utf-8") as f:
                            f.write(message)
                        print("💾 Đã lưu nội dung vào decoded_message.txt")
                else:
                    print("⚠️ Không tìm thấy thông điệp hoặc marker kết thúc.")
            except Exception as e:
                print("❌ Lỗi khi giải mã:", e)

        else:
            print("👋 Kết thúc chương trình.")


if __name__ == "__main__":
    main()
