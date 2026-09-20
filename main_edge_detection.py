import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from core.edge_detection import sobel_edge_optimized, canny_with_blur, hybrid_fuzzy_or_canny


def count_edges(edge_map):
    """
    Đếm số pixel biên (số điểm có giá trị > 0)
    """
    return int(np.sum(edge_map > 0))


def show_images(original, edge_map, method):
    """
    Hiển thị song song ảnh gốc và ảnh phát hiện cạnh
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Kết quả phát hiện cạnh - {method}")
    plt.imshow(edge_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_result(image, method_name):
    """
    Lưu ảnh ra file trong thư mục 'output_edges'
    """
    out_dir = "output_edges"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method_name}_edges.png")
    cv2.imwrite(out_path, image)
    print(f"[+] Ảnh đã được lưu tại: {out_path}")


def choose_image_file():
    """
    Hộp thoại chọn file ảnh để phát hiện cạnh
    """
    Tk().withdraw()  # Ẩn cửa sổ chính của Tkinter
    img_path = filedialog.askopenfilename(
        title="Chọn ảnh để phát hiện cạnh",
        filetypes=[("Ảnh", "*.png;*.jpg;*.jpeg;*.bmp"), ("Tất cả các tệp", "*.*")]
    )

    if not img_path:
        print("❌ Không chọn ảnh nào, thoát chương trình.")
        exit()

    if not os.path.exists(img_path):
        print("❌ Ảnh không tồn tại!")
        exit()

    return img_path


def run_edge_detection(return_result=False):
    img_path = choose_image_file()

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không đọc được ảnh!")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        print("\n===== MENU PHÁT HIỆN CẠNH =====")
        print("1. Sobel")
        print("2. Canny")
        print("3. Hybrid (Canny + Fuzzy)")
        print("4. Thoát")
        choice = input("Chọn chức năng: ").strip()

        if choice == '1':
            edge_map = sobel_edge_optimized(gray)
            method = "Sobel"
        elif choice == '2':
            edge_map = canny_with_blur(gray)
            method = "Canny"
        elif choice == '3':
            _, _, edge_map = hybrid_fuzzy_or_canny(gray)
            method = "Hybrid"
        elif choice == '4':
            print("👋 Thoát chương trình phát hiện cạnh.")
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ, thử lại!")
            continue

        num_edges = np.sum(edge_map > 0)
        print(f"[+] Phát hiện {num_edges} điểm biên ({method})")

        # 🔹 Sửa ở đây: truyền ảnh gốc, edge map và tên phương pháp
        show_images(img, edge_map, method)

        if return_result:
            return img_path, edge_map

        cont = input("Tiếp tục phát hiện cạnh khác? (y/n): ").strip().lower()
        if cont != 'y':
            break

    return None



if __name__ == "__main__":
    run_edge_detection()
