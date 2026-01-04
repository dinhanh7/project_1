def read_ifm_file(filename, shape):
    with open(filename, "r") as file:
        lines = file.readlines()    
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
def read_file_weight(filename, shape):
    with open(filename, "r") as file:
        lines = file.readlines()    
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

def write_ofm_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for h in range(H):          # Loop Channel trước (để khớp với genhex.py)
            for w in range(W):
                for c in range(C):
                    int_value = int(round(data[h, w, c]))
                    
                    # SỬA LẠI: Mask 32-bit và format 8 ký tự
                    hex_value = int_value & 0xFFFFFFFF 
                    file.write(f"{int_value}\n")
