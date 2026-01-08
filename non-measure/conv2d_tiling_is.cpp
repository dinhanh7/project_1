#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH BÀI TOÁN ---
#define INPUT_H 112
#define INPUT_W 112
#define INPUT_C 32
#define KERNEL_H 3
#define KERNEL_W 3
#define OUTPUT_F 1 
#define OUTPUT_H 112
#define OUTPUT_W 112
#define STRIDE 1
#define PADDING 1

// --- CẤU HÌNH PHẦN CỨNG ---
#define NUM_PE 48               
#define MACS_PER_PE 3           
#define BUFFER_SIZE_BYTES 144   // 144 bytes (Vừa đủ cho 16 channels x 9 phần tử)
#define PARALLEL_CHANNELS 16    

// --- MEMORY ---
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

int8_t buffer_ifm[BUFFER_SIZE_BYTES];   
int8_t buffer_weight[BUFFER_SIZE_BYTES]; 

void dram_init() {
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C);
    // Load IFM
    FILE* f_ifm = fopen("../params/ifm.txt", "r");
    if(f_ifm) {
        char line[64];
        
        for (int h = 0; h < INPUT_H; h++) {
            for (int w = 0; w < INPUT_W; w++) {
                for (int c = 0; c < INPUT_C; c++) {
                    
                    if (fgets(line, 64, f_ifm)) {
                        // Chuyển từ chuỗi sang số nguyên 
                        int val = atoi(line);
                        // Xử lý số nguyên có dấu 8-bit 
                        if (val > 0x7F) {
                            val -= 0x100;
                        }
                        // Công thức: index = h * (W * C) + w * C + c
                        // [h, w, c]
                        int idx = h * (INPUT_W * INPUT_C) + w * INPUT_C + c;
                        // Gán vào DRAM 
                        ifm_dram[idx] = (int8_t)val;
                    }
                }
            }
        }
        fclose(f_ifm);
    } else {
        printf("Error: Could not open ../params/ifm.txt\n");
        memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); 
    }
    // Weights
    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, 1);
    FILE* f_w = fopen("../params/weights.txt", "r");
    if(f_w) {
        char line[64];
        // WEITGHS = C->W->H->F
        for(int f=0; f<OUTPUT_F; f++)
            for(int h=0; h<KERNEL_H; h++)
                for(int w=0; w<KERNEL_W; w++)
                    for(int c=0; c<INPUT_C; c++)
                        if(fgets(line, 64, f_w)) {
                             int val = atoi(line);
                             if (val > 0x7F) val -= 0x100;
                             int idx = h*(KERNEL_W*INPUT_C*OUTPUT_F) + w*(INPUT_C*OUTPUT_F) + c*OUTPUT_F + f;
                             weight_dram[idx] = (int8_t)val;
                        }
        fclose(f_w);
    }

    // OFM (Dùng calloc để reset về 0 vì ta cần cộng dồn qua các pass)
    ofm_dram = (int32_t*)calloc(OUTPUT_H * OUTPUT_W * OUTPUT_F, sizeof(int32_t));
}
void write_dram_to_file() {
    FILE* f = fopen("../ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}
// INPUT SLIDING WINDOW LOGIC

// [INIT] Load toàn bộ 3x3 block (Chỉ chạy tại wo=0)
void dma_load_ifm_full(int ho, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                int hi = ho * STRIDE + kh - PADDING;
                int wi = 0 * STRIDE + kw - PADDING; // wo=0
                
                int8_t val = 0;
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
                }
                buffer_ifm[buffer_ptr++] = val;
            }
        }
    }
}

// [SLIDING] Shift trái buffer và chỉ load cột mới (Chạy tại wo > 0)
void dma_shift_and_load_ifm(int ho, int wo, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    
    // SHIFT BUFFER (Mô phỏng dịch chuyển thanh ghi)
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int base = i * 9; 
        // Dời cột 1 về 0, cột 2 về 1
        buffer_ifm[base + 0] = buffer_ifm[base + 1]; 
        buffer_ifm[base + 3] = buffer_ifm[base + 4]; 
        buffer_ifm[base + 6] = buffer_ifm[base + 7]; 
        buffer_ifm[base + 1] = buffer_ifm[base + 2]; 
        buffer_ifm[base + 4] = buffer_ifm[base + 5]; 
        buffer_ifm[base + 7] = buffer_ifm[base + 8]; 
    }

    // LOAD NEW COLUMN (Load cột thứ 3)
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;
        int base = i * 9;
        int wi = wo * STRIDE + 2 - PADDING; // Cột index 2 trong window

        for (int kh = 0; kh < KERNEL_H; kh++) { 
            int hi = ho * STRIDE + kh - PADDING;
            int8_t val = 0;
            if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
            }
            buffer_ifm[base + (kh * 3) + 2] = val; // Ghi vào vị trí cuối
        }
    }
}

// WEIGHT LOADING (Mô phỏng Tiling: Load lại liên tục)

// Hàm này sẽ được gọi TẠI MỖI PIXEL (WO) - Rất tốn kém băng thông
void dma_load_weights_per_pixel(int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                int w_idx = kh*(KERNEL_W*INPUT_C*OUTPUT_F) + kw*(INPUT_C*OUTPUT_F) + current_c*OUTPUT_F;
                buffer_weight[buffer_ptr++] = weight_dram[w_idx];
            }
        }
    }
}

// COMPUTE ENGINE & CONTROLLER

int32_t run_pe_array() {
    int32_t partial_sum = 0;
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        for (int k = 0; k < MACS_PER_PE; k++) {
            pe_acc += (int32_t)buffer_ifm[base_idx + k] * (int32_t)buffer_weight[base_idx + k];
        }
        partial_sum += pe_acc;
    }
    return partial_sum;
}

void run_simulation_hybrid() {
    printf("--- SIMULATION: TILING WEIGHTS + INPUT SLIDING WINDOW ---\n");
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;

    for (int ho = 0; ho < OUTPUT_H; ho++) {
        // Lưu ý: Đảo vòng lặp Pass ra ngoài Wo để giữ Buffer IFM cho Sliding Window
        for (int p = 0; p < num_passes; p++) {
            
            for (int wo = 0; wo < OUTPUT_W; wo++) {
                
                // WEIGHT LOADING (Kém hiệu quả - Theo yêu cầu)
                // Được gọi bên trong vòng lặp WO -> Load lại 112 lần mỗi hàng!
                dma_load_weights_per_pixel(p);

                // IFM LOADING (Hiệu quả - Sliding Window)
                if (wo == 0) {
                    dma_load_ifm_full(ho, p); // Init
                } else {
                    dma_shift_and_load_ifm(ho, wo, p); // Reuse & Shift
                }

                // COMPUTE
                int32_t res = run_pe_array();
                
                // Cộng dồn kết quả vào DRAM (vì Pass bị chia cắt)
                ofm_dram[ho * OUTPUT_W + wo] += res;
            }
        }
    }

    printf("--- SIMULATION COMPLETED ---\n");
}

void cleanup() { free(ifm_dram); free(weight_dram); free(ofm_dram); }

int main() {
    // Xóa file OFM cũ trước khi chạy
    remove("../ofm/ofm.txt");
    
    dram_init();
    run_simulation_hybrid();
    write_dram_to_file();
    cleanup();
    return 0;
}