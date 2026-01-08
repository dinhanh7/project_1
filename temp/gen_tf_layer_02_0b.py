import numpy as np
import tensorflow as tf
import argparse
import math

# BIAS_FRAC_BIT = 7
# --- BẮT ĐẦU ĐOẠN BỔ SUNG CÁC HÀM XỬ LÝ SCALE ---

# Hàm đọc file chứa 1 số thực (Scale của IFM, OFM)
def read_float_file(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        # Lấy phần tử cuối cùng nếu file có dạng "0.69..."
        val_str = content.split()[-1]
    return float(val_str)

# Hàm đọc file chứa nhiều số thực (Scale của Weight)
def read_float_array_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    # Lọc và lấy số thực từ mỗi dòng
    data = []
    for line in lines:
        val_str = line.strip().split()[-1]
        data.append(float(val_str))
    return np.array(data, dtype=np.float64)


# --- [BỔ SUNG 1] Hàm đọc giá trị ZP từ file ---
def read_zp_file(filename):
    try:
        with open(filename, "r") as file:
            # Đọc toàn bộ nội dung, ví dụ: "F2"
            content = file.read().strip()
            # Lấy chuỗi cuối cùng (F2) để tránh các ký tự thừa
            val_str = content.split()[-1]
            
        val = int(val_str)
        
        # Xử lý số âm 8-bit (nếu > 127 thì trừ 256)
        # Ví dụ: F2 (242) -> -14
        if val > 0x7F:
            val -= 0x100
        return val
    except Exception as e:
        print(f"Cảnh báo: Không đọc được file {filename}, dùng ZP=0. Lỗi: {e}")
        return 0
    
# Hàm đọc dữ liệu từ file HEX với thứ tự hàng → cột → channel → filter
def read_bias_file(filename, length):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Đọc dữ liệu Hex (Python tự hiểu 0xFF...F là số dương lớn)
    # Chúng ta dùng int(x, 16) sẽ ra số dương unsigned nếu x >= 0x80000000
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int64)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int64)
    # Xử lý số âm (Signed 32-bit conversion)
    # Nếu giá trị >= 2^31 (0x80000000), tức là số âm trong hệ bù 2 32-bit
    for i in range(len(data)):
        if data[i] >= 0x80000000: 
            data[i] -= 0x100000000  # Trừ đi 2^32 để về số âm

    # Ép kiểu về đúng int32 để đưa vào Model
    data = data.astype(np.int32)

    # LƯU Ý QUAN TRỌNG: 
    # Đã XÓA đoạn: data[j] = data[j] * (1 << BIAS_FRAC_BIT)
    # Bias là số nguyên cộng trực tiếp vào accumulator, không cần shift.

    return data.reshape((length,))

def read_hex_file_weight(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int16)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int16)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm

    H, W, C, F = shape
    reshaped_data = np.zeros((H, W, C, F), dtype=np.int16)
    index = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    reshaped_data[h, w, c, f] = data[index]
                    index += 1
    return reshaped_data

def read_hex_file(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int32)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int32)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    # Nếu giá trị lớn hơn 127, chúng ta sẽ chuyển thành số âm
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm
    H, W, C = shape
    reshaped_data = np.zeros((H, W, C), dtype=np.int32)
    index = 0
    for h in range(H):
        for w in range(W):
            for c in range(C):
                reshaped_data[h, w, c] = data[index]

                index += 1

    return reshaped_data

# Hàm ghi dữ liệu ra file HEX
# Sửa trong gen_tf_layer_01.py
# Sửa trong gen_tf_layer_01.py
def write_hex_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for h in range(H):          # Loop Channel trước (để khớp với genhex.py)
            for w in range(W):
                for c in range(C):
                    int_value = int(round(data[h, w, c]))
                    
                    # SỬA LẠI: Mask 32-bit và format 8 ký tự
                    hex_value = int_value & 0xFFFFFFFF 
                    file.write(f"{int_value}\n")
                    # file.write(f"{hex_value:08X}\n")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifm_height", type=int, required=True)
    parser.add_argument("--ifm_width", type=int, required=True)
    parser.add_argument("--ifm_channel", type=int, required=True)
    parser.add_argument("--weight_filter", type=int, required=True)
    parser.add_argument("--padding1", type=int, default=1)  # Padding P
    parser.add_argument("--stride1", type=int, default=1)   # Stride S
    args = parser.parse_args()

    # Tính kích thước OFM với padding và stride
    output_feature_height = (args.ifm_height - 3 + 2 * args.padding1) // args.stride1 + 1
    output_feature_width = (args.ifm_width - 3 + 2 * args.padding1) // args.stride1 + 1
    output_feature_channel = args.weight_filter

    # File paths cố định
    input_file = "HEX_IN/op006_CONV_2D_ifm_values.hex"
    weight_file = "HEX_IN/op006_CONV_2D_weight_values_0b.hex"
    output_file = "OFM/op006_CONV_2D_ofm.hex"

    
    # Đọc dữ liệu
    input_data = read_hex_file(input_file, (args.ifm_height, args.ifm_width, args.ifm_channel))
    weight_data_flat = read_hex_file_weight(weight_file, (3, 3, args.ifm_channel, args.weight_filter))
    weight_data = weight_data_flat.reshape(3, 3, args.ifm_channel, args.weight_filter)
    # Tính toán và thực hiện Padding thủ công
    if args.padding1 > 0:
        # Tính toán lượng cần pad theo chuẩn TensorFlow 'SAME'
        pad_h_total = max((output_feature_height - 1) * args.stride1 + 3 - args.ifm_height, 0)
        pad_w_total = max((output_feature_width - 1) * args.stride1 + 3 - args.ifm_width, 0)
        
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        print(f"Padding info: Top={pad_top}, Bottom={pad_bottom}, Left={pad_left}, Right={pad_right}")
        
        # Padding input data bằng giá trị ZP_IN
        input_data_padded = np.pad(
            input_data,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        input_data_padded = input_data

    # Cập nhật shape mới sau khi pad
    padded_height, padded_width, _ = input_data_padded.shape

    # Tạo mô hình với padding='valid' (vì đã pad tay rồi)
    input_layer = tf.keras.layers.Input(shape=(padded_height, padded_width, args.ifm_channel))
    conv_layer = tf.keras.layers.Conv2D(filters=args.weight_filter,
                                        kernel_size=(3, 3),
                                        strides=(args.stride1, args.stride1),
                                        padding='valid', # QUAN TRỌNG: Đổi thành VALID
                                        use_bias=False,
                                        activation=None)(input_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=conv_layer)
    model.layers[1].set_weights([weight_data.astype(np.float32)]) # <--- Bỏ bias_data đi
    # Dự đoán với input đã pad
    output_data = model.predict(input_data_padded.reshape(1, padded_height, padded_width, args.ifm_channel).astype(np.float32))
    output_data = output_data.reshape(output_feature_height, output_feature_width, output_feature_channel)
    write_hex_file(output_file, output_data)
    print(f"Kết quả raw Conv2D đã được ghi vào {output_file}")