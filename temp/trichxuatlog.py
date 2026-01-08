import re
import pandas as pd
import numpy as np

def parse_perf_logs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tách các block dựa trên dòng "Performance counter stats for"
    # Pattern này tìm tên lệnh (ví dụ: ./tl1)
    sections = re.split(r"Performance counter stats for '(\./.*?)':", content)
    
    data_list = []
    
    # Do dùng split, phần tử đầu tiên thường rỗng, nên ta duyệt theo bước nhảy 2
    # sections[i] là tên lệnh, sections[i+1] là nội dung log tương ứng
    for i in range(1, len(sections), 2):
        command = sections[i]
        block_content = sections[i+1]
        
        row_data = {'command': command}
        
        # Duyệt qua từng dòng trong block
        for line in block_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Xử lý dòng chứa thời gian (seconds time elapsed)
            # Định dạng: 0,022675880 seconds time elapsed
            time_match = re.search(r'([\d,\.]+)\s+seconds time elapsed', line)
            if time_match:
                val_str = time_match.group(1)
                # Chuyển đổi format 0,022 -> 0.022
                val = float(val_str.replace('.', '').replace(',', '.'))
                row_data['time_elapsed_seconds'] = val
                continue

            # Xử lý các dòng metrics (cache, cycles, instructions...)
            # Định dạng: 287.215      cpu_atom/cache-references/
            # Hoặc: <not counted>      cpu_atom/cache-references/
            metric_match = re.search(r'^([<>\w\s\.,]+)\s+(cpu_\w+/[^/]+/)', line)
            
            if metric_match:
                val_str = metric_match.group(1).strip()
                metric_name = metric_match.group(2).strip().replace('/', '_') # clean name
                
                if '<not counted>' in val_str:
                    row_data[metric_name] = np.nan
                else:
                    # Xử lý số: bỏ dấu chấm hàng nghìn, thay dấu phẩy thập phân bằng chấm
                    # Ví dụ: 27.687.235 -> 27687235;  2,49 -> 2.49
                    try:
                        clean_val = val_str.replace('.', '').replace(',', '.')
                        row_data[metric_name] = float(clean_val)
                    except ValueError:
                        row_data[metric_name] = np.nan

        data_list.append(row_data)

    # Tạo DataFrame
    df = pd.DataFrame(data_list)
    
    # Set command làm index để dễ so sánh
    df.set_index('command', inplace=True)
    
    return df

# Chạy thử với file logs.txt
df_results = parse_perf_logs('../non-measure/logs.txt')

# Hiển thị bảng kết quả
print("Bảng so sánh Performance:")
print(df_results)

# Xuất ra CSV nếu cần
df_results.to_csv('performance_comparison.csv')