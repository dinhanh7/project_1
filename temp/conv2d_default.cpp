#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH KÍCH THƯỚC (Theo shape [1, 3, 3, 32]) ---
#define INPUT_H 112
#define INPUT_W 112
#define INPUT_C 32  // Khớp với Channel = 32

// Weight Shape: [Filter, H, W, Channel] = [1, 3, 3, 32]
#define OUTPUT_F 1  // Filter
#define KERNEL_H 3  // H
#define KERNEL_W 3  // W
// INPUT_C ở trên là 32 (Channel)

#define OUTPUT_H 112
#define OUTPUT_W 112

#define STRIDE 1
#define PADDING 1   // Padding=1 để Input 112x112 -> Output 112x112 với Kernel 3x3

// --------------------------------------------------------

// Hàm đọc file IFM
int8_t* read_ifm_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    int total_elements = INPUT_H * INPUT_W * INPUT_C;
    int8_t* data = (int8_t*)malloc(total_elements * sizeof(int8_t));
    
    char line[64];
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        if (index >= total_elements) break;
        int val = atoi(line);
        data[index] = (int8_t)val;
        index++;
    }
    fclose(file);
    return data;
}

// Hàm đọc file Weight
// Cấu trúc file tuân theo shape [Filter, H, W, Channel]
// Nhưng lưu vào bộ nhớ theo layout [H, W, C, F] để tiện tính toán
int16_t* read_file_weight(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    int total_elements = KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F;
    int16_t* weights = (int16_t*)calloc(total_elements, sizeof(int16_t));
    
    char line[64];
    
    // Thứ tự Loop đọc file: Filter -> H -> W -> Channel
    for (int f = 0; f < OUTPUT_F; f++) {        // Dim 0: Filter (1)
        for (int h = 0; h < KERNEL_H; h++) {    // Dim 1: Height (3)
            for (int w = 0; w < KERNEL_W; w++) {// Dim 2: Width  (3)
                for (int c = 0; c < INPUT_C; c++) { // Dim 3: Channel (32)
                    
                    if (fgets(line, sizeof(line), file)) {
                        int val = atoi(line);
                        if (val > 0x7F) val -= 0x100;
                        
                        // Mapping vào bộ nhớ theo layout [H][W][C][F]
                        // Index = h*(W*C*F) + w*(C*F) + c*F + f
                        int dest_index = h * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                         w * (INPUT_C * OUTPUT_F) + 
                                         c * OUTPUT_F + 
                                         f;
                        
                        weights[dest_index] = (int16_t)val;
                    }
                }
            }
        }
    }
    fclose(file);
    return weights;
}

// Hàm tính toán Convolution: Trả về mảng int32
int32_t* conv2d(int8_t* ifm, int16_t* weights) {
    int total_elements = OUTPUT_H * OUTPUT_W * OUTPUT_F;
    int32_t* ofm_data = (int32_t*)malloc(total_elements * sizeof(int32_t));

    if (!ofm_data) {
        printf("Error: Memory allocation failed for OFM\n");
        exit(1);
    }

    // Loop Output: H -> W -> F
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        for (int wo = 0; wo < OUTPUT_W; wo++) {
            for (int fo = 0; fo < OUTPUT_F; fo++) {
                
                int32_t accumulator = 0; 

                // Loop Kernel: H -> W -> Input Channel
                for (int kh = 0; kh < KERNEL_H; kh++) {
                    for (int kw = 0; kw < KERNEL_W; kw++) {
                        
                        int hi = ho * STRIDE + kh - PADDING;
                        int wi = wo * STRIDE + kw - PADDING;

                        if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                            for (int ci = 0; ci < INPUT_C; ci++) {
                                // Lấy IFM tại [hi, wi, ci]
                                int ifm_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + ci;
                                int8_t ifm_val = ifm[ifm_idx];

                                // Lấy Weight tại [kh, kw, ci, fo]
                                int w_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                            kw * (INPUT_C * OUTPUT_F) + 
                                            ci * OUTPUT_F + 
                                            fo;
                                int16_t w_val = weights[w_idx];

                                accumulator += (int32_t)ifm_val * (int32_t)w_val;
                            }
                        }
                    }
                }

                // Lưu kết quả [ho, wo, fo]
                int out_idx = ho * (OUTPUT_W * OUTPUT_F) + wo * OUTPUT_F + fo;
                ofm_data[out_idx] = accumulator;
            }
        }
    }
    return ofm_data;
}

// Hàm ghi file OFM
void write_ofm_file(const char* filename, int32_t* data) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file %s for writing\n", filename);
        exit(1);
    }

    int total_elements = OUTPUT_H * OUTPUT_W * OUTPUT_F;
    for (int i = 0; i < total_elements; i++) {
        fprintf(file, "%d\n", data[i]);
    }

    fclose(file);
}

int main() {
    printf("Starting C convolution...\n");
    printf("Config: Input[%d,%d,%d], Kernel[%d,%d], Output[%d,%d,%d]\n", 
           INPUT_H, INPUT_W, INPUT_C, KERNEL_H, KERNEL_W, OUTPUT_H, OUTPUT_W, OUTPUT_F);

    // Đọc dữ liệu
    int8_t* ifm_data = read_ifm_file("params/ifm.txt");
    int16_t* weight_data = read_file_weight("params/weights.txt");

    // Tính toán
    printf("Computing...\n");
    int32_t* ofm_data = conv2d(ifm_data, weight_data);

    // Ghi file
    printf("Writing OFM...\n");
    write_ofm_file("ofm/ofm.txt", ofm_data);

    // Giải phóng
    free(ifm_data);
    free(weight_data);
    free(ofm_data);

    printf("Done.\n");
    return 0;
}