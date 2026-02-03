import subprocess
import matplotlib.pyplot as plt
import re
import pandas as pd
import os

# --- CẤU HÌNH ---
# Danh sách các file nguồn và tên file thực thi tương ứng
architectures = {
    "ISC": "config_conv2d_tiling_is.cpp",
    "WS": "config_conv2d_tiling_ws.cpp",
    "WSIS": "config_conv2d_tiling_ws_is.cpp",
    "TL": "config_conv2d_tiling.cpp"
}

# Tham số Shape cố định (IH, IW, IC, KH, KW, OF, OH, OW, S, P)
SHAPE_ARGS = ["112", "112", "32", "3", "3", "1", "112", "112", "1", "1"]

# Dải survey: Số kênh song song (Parallel Channels)
# Sẽ tính ra NUM_PE và BUFFER tương ứng
target_channels = [1, 2, 4, 8, 16, 32, 48] 

# --- BƯỚC 1: BIÊN DỊCH (COMPILE) ---
print("--- Đang biên dịch các kiến trúc ---")
for name, source in architectures.items():
    compile_cmd = ["g++", source, "-o", name.lower()]
    subprocess.run(compile_cmd, check=True)
    print(f"Đã biên dịch {name}")

# --- BƯỚC 2: CHẠY KHẢO SÁT ---
all_results = []

for name in architectures.keys():
    print(f"\n--- Đang khảo sát kiến trúc: {name} ---")
    executable = f"./{name.lower()}"
    
    for ch in target_channels:
        # Tính toán tham số HW
        total_macs = ch * 9
        num_pe = int(total_macs / 3) # Giả sử mỗi PE có 3 MACs, bạn có thể đổi tùy ý
        macs_per_pe = 3
        buffer_size = total_macs
        
        cmd = [executable] + SHAPE_ARGS + [str(num_pe), str(macs_per_pe), str(buffer_size)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            
            # Bắt kết quả từ dòng SURVEY_RESULT,dma,compute,total
            match = re.search(r"SURVEY_RESULT,(\d+),(\d+),(\d+)", output)
            if match:
                dma = int(match.group(1))
                comp = int(match.group(2))
                total = int(match.group(3))
                
                all_results.append({
                    "Architecture": name,
                    "Parallel_Channels": ch,
                    "Total_MACs": total_macs,
                    "DMA_Cycles": dma,
                    "Compute_Cycles": comp,
                    "Total_Cycles": total
                })
                print(f"Channels {ch:2d}: Total Cycles = {total}")
        except Exception as e:
            print(f"Lỗi khi chạy {name} với {ch} channels: {e}")

# --- BƯỚC 3: LƯU DỮ LIỆU & VẼ BIỂU ĐỒ ---
df = pd.DataFrame(all_results)
df.to_csv("master_survey_results.csv", index=False)



plt.figure(figsize=(12, 7))

for name in architectures.keys():
    data = df[df["Architecture"] == name]
    plt.plot(data["Total_MACs"], data["Total_Cycles"], marker='o', label=f'Dataflow: {name}', linewidth=2)

plt.title("So sánh hiệu năng giữa các kiến trúc Dataflow (Input Channel=32)")
plt.xlabel("Tài nguyên phần cứng (Tổng số MACs)")
plt.ylabel("Tổng số chu kỳ máy (Cycles - Càng thấp càng tốt)")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Đánh dấu điểm bão hòa của Input (32 ch * 9 = 288 MACs)
plt.axvline(x=288, color='red', linestyle='--', alpha=0.6, label='Input Channel Limit (32)')

plt.savefig("comparison_report.png")
print("\n--- Xong! Đã lưu 'master_survey_results.csv' và 'comparison_report.png' ---")