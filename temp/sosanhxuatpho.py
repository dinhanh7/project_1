import matplotlib.pyplot as plt
import os

def compare_hex_files_larger_points(file1, file2):
    print(f"\n--- Đang so sánh ---")
    print(f"File 1 (Golden): {file1}")
    print(f"File 2 (OFM)   : {file2}")
    print("Đang đọc và xử lý dữ liệu...")
    
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file hoặc đường dẫn sai!")
        return

    max_len = max(len(lines1), len(lines2))
    diff_indices = []
    diff_values = []

    for i in range(max_len):
        str1 = lines1[i].strip() if i < len(lines1) else None
        str2 = lines2[i].strip() if i < len(lines2) else None
        try:
            val1 = int(str1, 16) if str1 else 0
            val2 = int(str2, 16) if str2 else 0
            diff = abs(val1 - val2)
            if diff > 0:
                diff_indices.append(i + 1)
                diff_values.append(diff)
        except ValueError:
            pass

    total_diff = len(diff_values)
    print(f"Xử lý xong. Tổng số lỗi: {total_diff}")

    if total_diff > 0:
        plt.figure(figsize=(14, 6), dpi=100)

        plt.scatter(diff_indices, diff_values,
                    s=15,
                    marker='o',
                    alpha=0.3,
                    label='Điểm sai lệch')

        name1 = os.path.basename(file1)
        name2 = os.path.basename(file2)

        plt.title(f'So sánh: {name1} vs {name2}\nTổng số mẫu sai khác: {total_diff:,} mẫu')
        plt.xlabel('Vị trí dòng (Line Index)')
        plt.ylabel('Độ lệch tuyệt đối (Absolute Error)')
        plt.grid(True, linestyle='--', alpha=0.5)

        if diff_values:
            plt.ylim(0, max(diff_values) * 1.1)

        plt.tight_layout()
        print("Đang hiển thị biểu đồ...")
        plt.show()
    else:
        print("Kết quả: Hai file GIỐNG HỆT nhau! (Không có lỗi nào)")

def find_file_with_prefix(folder_path, prefix):
    try:
        if not os.path.exists(folder_path):
            print(f"Lỗi: Thư mục '{folder_path}' không tồn tại.")
            return None

        files = os.listdir(folder_path)
        for f in files:
            if f.startswith(prefix):
                return os.path.join(folder_path, f)
        return None
    except Exception as e:
        print(f"Lỗi khi tìm kiếm: {e}")
        return None

# ===================== MAIN =====================
if __name__ == "__main__":

    GOLDEN_DIR = "../golden_output"
    OFM_DIR = "../ofm/"

    search_prefix = "ofm"
    print(f"-> Đang tìm file bắt đầu bằng '{search_prefix}'...")

    golden_file = find_file_with_prefix(GOLDEN_DIR, search_prefix)
    ofm_file = find_file_with_prefix(OFM_DIR, search_prefix)

    if golden_file and ofm_file:
        compare_hex_files_larger_points(golden_file, ofm_file)
    else:
        print("\nKHÔNG TÌM THẤY FILE PHÙ HỢP!")
        if not golden_file:
            print(f"- Không thấy file nào bắt đầu bằng 'ofm' trong {GOLDEN_DIR}")
        if not ofm_file:
            print(f"- Không thấy file nào bắt đầu bằng 'ofm' trong {OFM_DIR}")
