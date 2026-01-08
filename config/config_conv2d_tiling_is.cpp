// ./isc 112 112 32 3 3 1 112 112 1 1 48 3 144
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH BÀI TOÁN ---
// #define INPUT_H 112
// #define INPUT_W 112
// #define INPUT_C 32
// #define KERNEL_H 3
// #define KERNEL_W 3
// #define OUTPUT_F 1 
// #define OUTPUT_H 112
// #define OUTPUT_W 112
// #define STRIDE 1
// #define PADDING 1
int INPUT_H, INPUT_W, INPUT_C;
int KERNEL_H, KERNEL_W;
int OUTPUT_F, OUTPUT_H, OUTPUT_W;
int STRIDE, PADDING;

// --- CẤU HÌNH PHẦN CỨNG ---
// #define NUM_PE 48               
// #define MACS_PER_PE 3           
// #define BUFFER_SIZE_BYTES 144   // 48 PE * 3 inputs * 1 byte
// #define PARALLEL_CHANNELS 16    // Số channel xử lý song song
int NUM_PE, MACS_PER_PE, BUFFER_SIZE_BYTES;
int PARALLEL_CHANNELS;
// --- CẤU HÌNH HIỆU NĂNG ---
#define SYSTEM_FREQ_MHZ 100.0   
#define DRAM_BUS_WIDTH_BYTES 8  
#define PE_COMPUTE_CYCLES 1     

unsigned long long total_dma_cycles = 0;
unsigned long long total_compute_cycles = 0;

// --- MEMORY ---
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

int8_t* buffer_ifm;   
int8_t* buffer_weight;


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
    // Latency: Full Load 144 bytes
    total_dma_cycles += (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
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
    int bytes_loaded = 0;
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
            bytes_loaded++;
        }
    }
    // Latency: Partial Load 48 bytes (Nhanh gấp 3 lần full load)
    total_dma_cycles += (bytes_loaded + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
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
    // Latency: Luôn load 144 bytes mỗi lần gọi
    total_dma_cycles += (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
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
    total_compute_cycles += PE_COMPUTE_CYCLES;
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

    // REPORT
    unsigned long long total_cycles = total_dma_cycles + total_compute_cycles;
    double total_time_ms = (double)total_cycles / (SYSTEM_FREQ_MHZ * 1000.0);
    
    printf("\n--- PERFORMANCE REPORT (Hybrid) ---\n");
    printf("Total Cycles: %llu\n", total_cycles);
    printf("  - DMA Cycles:     %llu (High Weight load, Low IFM load)\n", total_dma_cycles);
    printf("  - Compute Cycles: %llu\n", total_compute_cycles);
    printf("Estimated Time: %.4f ms\n", total_time_ms);
    printf("-----------------------------------\n");
}

void cleanup() { free(ifm_dram); free(weight_dram); free(ofm_dram); }

// int main() {
//     dram_init();
//     run_simulation_hybrid();
//     write_dram_to_file();
//     cleanup();
//     return 0;
// }
int main(int argc, char *argv[]) {
    // Kiểm tra đủ tham số (13 tham số + 1 tên file = 14)
    if (argc < 14) {
        printf("Usage: %s IH IW IC KH KW OF OH OW S P NPE MAC BUF\n", argv[0]);
        return -1;
    }

    // Gán giá trị từ Terminal
    INPUT_H = atoi(argv[1]);
    INPUT_W = atoi(argv[2]);
    INPUT_C = atoi(argv[3]);
    KERNEL_H = atoi(argv[4]);
    KERNEL_W = atoi(argv[5]);
    OUTPUT_F = atoi(argv[6]);
    OUTPUT_H = atoi(argv[7]);
    OUTPUT_W = atoi(argv[8]);
    STRIDE = atoi(argv[9]);
    PADDING = atoi(argv[10]);
    NUM_PE = atoi(argv[11]);
    MACS_PER_PE = atoi(argv[12]);
    BUFFER_SIZE_BYTES = atoi(argv[13]);

    // Tự động tính PARALLEL_CHANNELS
    // Logic: (Tổng số MACs của mảng PE) / (Kích thước 1 kernel)
    // Ví dụ: (48 * 3) / (3 * 3) = 16
    int kernel_size = KERNEL_H * KERNEL_W;
    if (kernel_size > 0) {
        PARALLEL_CHANNELS = (NUM_PE * MACS_PER_PE) / kernel_size;
    } else {
        PARALLEL_CHANNELS = 1;
    }
    printf("--- Configuration ---\n");
    printf("Parallel Channels: %d\n", PARALLEL_CHANNELS);
    printf("Buffer Size: %d bytes\n", BUFFER_SIZE_BYTES);

    // Cấp phát bộ nhớ động cho Buffer
    buffer_ifm = (int8_t*)malloc(BUFFER_SIZE_BYTES * sizeof(int8_t));
    buffer_weight = (int8_t*)malloc(BUFFER_SIZE_BYTES * sizeof(int8_t));

    if (!buffer_ifm || !buffer_weight) {
        printf("Error: Malloc failed for buffers\n");
        return -1;
    }

    // Chạy quy trình mô phỏng cũ
    dram_init();
    run_simulation_hybrid(); // Hàm chạy chính của file này
    write_dram_to_file();
    
    // Dọn dẹp bộ nhớ
    free(buffer_ifm);
    free(buffer_weight);
    cleanup(); // Dọn dẹp các DRAM

    return 0;
}