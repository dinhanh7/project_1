import subprocess
import matplotlib.pyplot as plt
import re
import pandas as pd
import os

# --- CẤU HÌNH ---
architectures = {
    "ISC": "config_conv2d_tiling_is.cpp",
    "WS": "config_conv2d_tiling_ws.cpp",
    "WSIS": "config_conv2d_tiling_ws_is.cpp",
    "TL": "config_conv2d_tiling.cpp"
}

SHAPE_ARGS = ["112", "112", "32", "3", "3", "1", "112", "112", "1", "1"]
target_channels = [1, 2, 4, 8, 16, 32, 48] 

# --- HÀM PARSE PERF (Chuyên biệt cho format cpu_core/...) ---
def parse_perf_text_output(stderr_text):
    data = {
        "cache-references": 0,
        "cache-misses": 0,
        "cycles": 0,
        "instructions": 0,
        "branches": 0,
        "seconds": 0.0
    }
    
    # Regex bắt các dòng metric của cpu_core (xử lý dấu chấm hàng nghìn)
    metric_pattern = re.compile(r"^\s*([\d\.]+)\s+cpu_core\/([\w\-]+)\/")
    # Regex bắt thời gian (xử lý dấu phẩy thập phân)
    time_pattern = re.compile(r"([\d\,]+)\s+seconds time elapsed")

    for line in stderr_text.splitlines():
        # Xử lý Metric
        match_metric = metric_pattern.search(line)
        if match_metric:
            val_str = match_metric.group(1).replace('.', '') # 1.000 -> 1000
            metric_name = match_metric.group(2)
            try:
                val = int(val_str)
                if "cache-references" in metric_name: data["cache-references"] = val
                elif "cache-misses" in metric_name: data["cache-misses"] = val
                elif "cycles" in metric_name: data["cycles"] = val
                elif "instructions" in metric_name: data["instructions"] = val
                elif "branches" in metric_name: data["branches"] = val
            except ValueError: pass

        # Xử lý Seconds
        match_time = time_pattern.search(line)
        if match_time:
            time_str = match_time.group(1).replace(',', '.') # 0,001 -> 0.001
            try: data["seconds"] = float(time_str)
            except ValueError: pass
            
    return data

# --- BƯỚC 1: BIÊN DỊCH ---
print("--- Đang biên dịch các kiến trúc ---")
for name, source in architectures.items():
    compile_cmd = ["g++", source, "-o", name.lower()]
    subprocess.run(compile_cmd, check=True)
    print(f"Đã biên dịch {name}")

# --- BƯỚC 2: CHẠY KHẢO SÁT ---
all_results = []
print("\n--- Bắt đầu chạy Benchmark (Cần SUDO) ---")

for name in architectures.keys():
    executable = f"./{name.lower()}"
    
    for ch in target_channels:
        total_macs = ch * 9
        num_pe = int(total_macs / 3)
        macs_per_pe = 3
        buffer_size = total_macs
        
        # Lệnh Perf bắt sự kiện cpu_core
        perf_cmd = [
            "sudo", "taskset", "-c", "0-7", 
            "perf", "stat", 
            "-e", "cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/cycles/,cpu_core/instructions/,cpu_core/branches/"
        ]
        
        app_cmd = [executable] + SHAPE_ARGS + [str(num_pe), str(macs_per_pe), str(buffer_size)]
        full_cmd = perf_cmd + app_cmd
        
        try:
            result = subprocess.run(full_cmd, capture_output=True, text=True)
            
            # 1. Lấy kết quả từ code C++ (NHƯ CODE CŨ CỦA BẠN)
            match_cpp = re.search(r"SURVEY_RESULT,(\d+),(\d+),(\d+)", result.stdout)
            
            # 2. Lấy kết quả từ Perf
            perf_data = parse_perf_text_output(result.stderr)
            
            if match_cpp:
                dma = int(match_cpp.group(1))
                comp = int(match_cpp.group(2))
                total_hw = int(match_cpp.group(3))
                
                # Tính toán số liệu Perf
                cache_refs = perf_data["cache-references"]
                cache_miss = perf_data["cache-misses"]
                miss_rate = (cache_miss / cache_refs * 100) if cache_refs > 0 else 0.0

                # --- ĐÂY LÀ PHẦN QUAN TRỌNG: GỘP CẢ 2 ---
                record = {
                    "Architecture": name,
                    "Parallel_Channels": ch,
                    "Total_MACs": total_macs,
                    "NUM_PE": num_pe,                 # <--- THÊM DÒNG NÀY
                    "MACS_PER_PE": macs_per_pe,       # <--- THÊM DÒNG NÀY
                    "BUFFER_SIZE_BYTES": buffer_size, # <--- THÊM DÒNG NÀY
                    # === NHÓM 1: DỮ LIỆU TỪ CODE CŨ (Simulation) ===
                    "DMA_Cycles": dma,        # <--- Đã thêm lại
                    "Compute_Cycles": comp,   # <--- Đã thêm lại
                    "Total_Cycles": total_hw, # <--- Đã thêm lại (Simulated)
                    
                    # === NHÓM 2: DỮ LIỆU TỪ PERF (Real Hardware) ===
                    "cpu_core_cache": cache_refs,
                    "references_cpu_core_cache": cache_refs,
                    "misses_cpu_core_miss": cache_miss,
                    "percentagecpu_core": round(miss_rate, 2),
                    "cycles_cpu_core": perf_data["cycles"],
                    "instructions_cpu_core": perf_data["instructions"],
                    "branches_time_elapsed": perf_data["branches"],
                    "seconds": perf_data["seconds"]
                }
                
                all_results.append(record)
                print(f"[{name}] Ch={ch:2d} | Sim_Cycles={total_hw} | Perf_Sec={perf_data['seconds']:.5f}")
            else:
                print(f"Lỗi: Không tìm thấy SURVEY_RESULT cho {name} (ch={ch})")
                
        except Exception as e:
            print(f"Exception tại {name} ch={ch}: {e}")

# --- BƯỚC 3: LƯU VÀ VẼ ---
if all_results:
    df = pd.DataFrame(all_results)
    
    # Lưu CSV đầy đủ
    df.to_csv("master_survey_results_FULL.csv", index=False)
    print("\n--- Đã lưu 'master_survey_results_FULL.csv' ---")

    # --- VẼ BIỂU ĐỒ 1: THEO CODE CŨ (Simulation Cycles) ---
    plt.figure(figsize=(12, 7))
    for name in architectures.keys():
        data = df[df["Architecture"] == name]
        plt.plot(data["Total_MACs"], data["Total_Cycles"], marker='o', label=f'Sim: {name}', linewidth=2)

    plt.title("So sánh hiệu năng Simulation (Total Cycles)")
    plt.xlabel("Tài nguyên phần cứng (Tổng số MACs)")
    plt.ylabel("Tổng số chu kỳ máy (Simulated)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.axvline(x=288, color='red', linestyle='--', alpha=0.6, label='Input Channel Limit')
    plt.savefig("comparison_report_sim.png")

    # --- VẼ BIỂU ĐỒ 2: THEO PERF DATA (Real Time) ---
    plt.figure(figsize=(12, 7))
    for name in architectures.keys():
        data = df[df["Architecture"] == name]
        # Vẽ thời gian thực (Seconds) thay vì Cycles
        plt.plot(data["Total_MACs"], data["seconds"], marker='x', linestyle='--', label=f'Real: {name}')
        
    plt.title("So sánh hiệu năng Thực tế trên Jetson (Seconds)")
    plt.xlabel("Total MACs")
    plt.ylabel("Execution Time (seconds)")
    plt.xscale('log')
    # plt.yscale('log') # Thường thời gian thực tuyến tính dễ nhìn hơn, bỏ log y nếu muốn
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("comparison_report_real.png")
    
    print("--- Đã vẽ xong 2 biểu đồ (Sim và Real) ---")
else:
    print("Không có dữ liệu.")