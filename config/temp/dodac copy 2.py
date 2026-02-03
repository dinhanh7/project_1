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

# --- HÀM PARSE CHUYÊN BIỆT CHO FORMAT CỦA BẠN ---
def parse_perf_text_output(stderr_text):
    data = {
        "cache-references": 0,
        "cache-misses": 0,
        "cycles": 0,
        "instructions": 0,
        "branches": 0,
        "seconds": 0.0
    }
    
    # Debug: In ra để kiểm tra nếu cần
    # print(f"\n--- DEBUG RAW PERF ---\n{stderr_text}\n----------------------")

    # 1. Regex bắt các dòng metric của cpu_core
    # Pattern giải thích:
    #   \s* : Khoảng trắng đầu dòng
    #   ([\d\.]+)   : Nhóm 1 - Giá trị số (chấp nhận cả số và dấu chấm hàng nghìn)
    #   \s+         : Khoảng trắng
    #   cpu_core\/  : Tìm đúng prefix "cpu_core/" (bỏ qua cpu_atom)
    #   ([\w\-]+)   : Nhóm 2 - Tên metric (cache-misses, cycles...)
    metric_pattern = re.compile(r"^\s*([\d\.]+)\s+cpu_core\/([\w\-]+)\/")
    
    # 2. Regex bắt thời gian (xử lý dấu phẩy)
    # Pattern: Tìm số có dấu phẩy đứng trước chữ "seconds time elapsed"
    time_pattern = re.compile(r"([\d\,]+)\s+seconds time elapsed")

    for line in stderr_text.splitlines():
        # A. Xử lý các chỉ số (Cycles, Cache...)
        match_metric = metric_pattern.search(line)
        if match_metric:
            val_str = match_metric.group(1)   # Ví dụ: "1.061.216"
            metric_name = match_metric.group(2) # Ví dụ: "cycles"
            
            # Xóa dấu chấm hàng nghìn để thành số int chuẩn (1061216)
            clean_val = val_str.replace('.', '')
            
            try:
                val = int(clean_val)
                if "cache-references" in metric_name:
                    data["cache-references"] = val
                elif "cache-misses" in metric_name:
                    data["cache-misses"] = val
                elif "cycles" in metric_name:
                    data["cycles"] = val
                elif "instructions" in metric_name:
                    data["instructions"] = val
                elif "branches" in metric_name:
                    data["branches"] = val
            except ValueError:
                pass

        # B. Xử lý thời gian (Seconds)
        match_time = time_pattern.search(line)
        if match_time:
            time_str = match_time.group(1) # Ví dụ: "0,000815344"
            # Thay dấu phẩy thành dấu chấm để Python hiểu là float
            clean_time = time_str.replace(',', '.')
            try:
                data["seconds"] = float(clean_time)
            except ValueError:
                pass
            
    return data

# --- BƯỚC 1: BIÊN DỊCH ---
print("--- Đang biên dịch các kiến trúc ---")
for name, source in architectures.items():
    compile_cmd = ["g++", source, "-o", name.lower()]
    subprocess.run(compile_cmd, check=True)
    print(f"Đã biên dịch {name}")

# --- BƯỚC 2: CHẠY KHẢO SÁT ---
all_results = []
print("\n--- Bắt đầu chạy Benchmark (Lưu ý chạy với SUDO) ---")

for name in architectures.keys():
    executable = f"./{name.lower()}"
    
    for ch in target_channels:
        # Tính toán tham số
        total_macs = ch * 9
        num_pe = int(total_macs / 3)
        macs_per_pe = 3
        buffer_size = total_macs
        
        # Lệnh Perf (BỎ cờ -x để lấy output text gốc như bạn cung cấp)
        perf_cmd = [
            "sudo", "taskset", "-c", "0-7", 
            "perf", "stat", 
            # KHÔNG DÙNG "-x ,", để perf xuất text mặc định như bạn muốn
            "-e", "cpu_core/cache-references/,cpu_core/cache-misses/,cpu_core/cycles/,cpu_core/instructions/,cpu_core/branches/"
        ]
        
        app_cmd = [executable] + SHAPE_ARGS + [str(num_pe), str(macs_per_pe), str(buffer_size)]
        full_cmd = perf_cmd + app_cmd
        
        try:
            # Chạy lệnh
            result = subprocess.run(full_cmd, capture_output=True, text=True)
            
            # Parse STDOUT (Kết quả C++)
            match_cpp = re.search(r"SURVEY_RESULT,(\d+),(\d+),(\d+)", result.stdout)
            
            # Parse STDERR (Kết quả Perf - dùng hàm mới)
            perf_data = parse_perf_text_output(result.stderr)
            
            if match_cpp:
                # Lấy dữ liệu mô phỏng
                dma = int(match_cpp.group(1))
                comp = int(match_cpp.group(2))
                total_hw = int(match_cpp.group(3))
                
                # Lấy dữ liệu thực tế
                cache_refs = perf_data["cache-references"]
                cache_miss = perf_data["cache-misses"]
                miss_rate = (cache_miss / cache_refs * 100) if cache_refs > 0 else 0.0

                record = {
                    "Architecture": name,
                    "Parallel_Channels": ch,
                    "Total_MACs": total_macs,
                    
                    # Simulation Data
                    "Sim_Total_Cycles": total_hw,
                    
                    # Real Perf Data (Tên cột như bạn yêu cầu)
                    "cpu_core_cache": cache_refs,             # references_cpu_core_cache (gộp ý nghĩa)
                    "references_cpu_core_cache": cache_refs,
                    "misses_cpu_core_miss": cache_miss,
                    "percentagecpu_core": round(miss_rate, 2),
                    "cycles_cpu_core": perf_data["cycles"],
                    "instructions_cpu_core": perf_data["instructions"],
                    "branches_time_elapsed": perf_data["branches"], # Tên cột hơi lạ, nhưng tôi map đúng vào branches
                    "seconds": perf_data["seconds"]
                }
                
                all_results.append(record)
                print(f"[{name}] Ch={ch:2d} | Time={perf_data['seconds']:.6f}s | Miss={miss_rate:.2f}% | Cycles={perf_data['cycles']}")
            else:
                # Nếu không bắt được kết quả C++, in lỗi
                print(f"Lỗi: Không tìm thấy SURVEY_RESULT cho {name} (ch={ch})")
                
        except Exception as e:
            print(f"Exception: {e}")

# --- BƯỚC 3: LƯU KẾT QUẢ ---
if all_results:
    df = pd.DataFrame(all_results)
    # Sắp xếp lại cột cho dễ nhìn theo ý bạn
    cols = [
        "Architecture", "Parallel_Channels", "Total_MACs", 
        "cpu_core_cache", "references_cpu_core_cache", "misses_cpu_core_miss", 
        "percentagecpu_core", "cycles_cpu_core", "instructions_cpu_core", 
        "branches_time_elapsed", "seconds", "Sim_Total_Cycles"
    ]
    # Chỉ giữ lại các cột có trong danh sách trên (nếu thiếu cột nào pandas sẽ báo lỗi, nên ta lọc intersection)
    cols = [c for c in cols if c in df.columns]
    
    df = df[cols]
    df.to_csv("master_survey_results_perf_fixed.csv", index=False)
    print("\n--- Xong! Đã lưu file 'master_survey_results_perf_fixed.csv' ---")
else:
    print("Không thu thập được dữ liệu nào.")