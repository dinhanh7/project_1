#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

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
#define BUFFER_SIZE_BYTES 144   // 1152 bit = 144 bytes
#define PARALLEL_CHANNELS 16    // 16 channels song song

// --- CẤU HÌNH HIỆU NĂNG ---
#define SYSTEM_FREQ_MHZ 100.0   
#define DRAM_BUS_WIDTH_BYTES 8  
#define PE_COMPUTE_CYCLES 1     

// --- MÔ PHỎNG BỘ NHỚ ---
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

int8_t buffer_ifm[BUFFER_SIZE_BYTES];   
int8_t buffer_weight[BUFFER_SIZE_BYTES]; 

// Biến đếm số bytes DMA load
uint64_t total_weight_bytes_loaded = 0;
uint64_t total_ifm_bytes_loaded = 0;

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
        printf("Error: Could not open params/ifm.txt\n");
        memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); 
    }

    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, 1);
    FILE* f_w = fopen("../params/weights.txt", "r");
    if(f_w) {
        char line[64];
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

    ofm_dram = (int32_t*)calloc(OUTPUT_H * OUTPUT_W * OUTPUT_F, sizeof(int32_t));
}

// ==================================================================================
// 1. CÁC HÀM DMA (Weight, IFM Init, IFM Shift)
// ==================================================================================

// Load Weight (Weight Stationary - Chỉ chạy đầu Pass)
void dma_load_weights(int pass_idx) {
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
    
    // Đếm số bytes đã load (144 bytes mỗi pass)
    total_weight_bytes_loaded += 144;
}

// IFM INIT: Load toàn bộ 3x3 block (Chạy tại điểm đầu tiên của mỗi hàng: wo=0)
// Tương ứng với "Khung màu Đỏ"
void dma_load_ifm_init(int ho, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                int hi = ho * STRIDE + kh - PADDING;
                int wi = 0 * STRIDE + kw - PADDING; // wo = 0
                
                int8_t val = 0;
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
                }
                buffer_ifm[buffer_ptr++] = val;
            }
        }
    }
    
    // Đếm số bytes đã load (144 bytes - toàn bộ 3x3 window cho 16 channels)
    total_ifm_bytes_loaded += 144;
}

// IFM SHIFT & LOAD: Dịch buffer và chỉ load cột mới
// Tương ứng với việc chuyển sang "Khung màu Tím"
void dma_shift_and_load_col(int ho, int wo, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    
    // 1. SHIFT PHASE (Dịch chuyển dữ liệu trong SRAM/Register)
    // Cấu trúc buffer cho mỗi channel (3x3):
    // [0] [1] [2]  --> Cột 0: idx 0,3,6
    // [3] [4] [5]  --> Cột 1: idx 1,4,7
    // [6] [7] [8]  --> Cột 2: idx 2,5,8
    
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int base = i * 9; 
        
        // Cột 1 cũ -> Cột 0 mới
        buffer_ifm[base + 0] = buffer_ifm[base + 1]; 
        buffer_ifm[base + 3] = buffer_ifm[base + 4]; 
        buffer_ifm[base + 6] = buffer_ifm[base + 7]; 

        // Cột 2 cũ -> Cột 1 mới
        buffer_ifm[base + 1] = buffer_ifm[base + 2]; 
        buffer_ifm[base + 4] = buffer_ifm[base + 5]; 
        buffer_ifm[base + 7] = buffer_ifm[base + 8]; 
    }

    // 2. LOAD PHASE (Chỉ load cột thứ 3 từ DRAM)
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;
        int base = i * 9;

        // Tính tọa độ cột mới bên phải cùng của cửa sổ 3x3
        // Cột mới là index 2 trong kernel window (0, 1, [2])
        int wi = wo * STRIDE + 2 - PADDING;

        for (int kh = 0; kh < KERNEL_H; kh++) { // 3 dòng của cột mới
            int hi = ho * STRIDE + kh - PADDING;
            
            int8_t val = 0;
            if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
            }
            
            // Ghi vào vị trí cột 2 (Index 2, 5, 8)
            buffer_ifm[base + (kh * 3) + 2] = val;
        }
    }
    
    // Đếm số bytes đã load (48 bytes - chỉ cột mới: 16 channels × 3 rows)
    total_ifm_bytes_loaded += 48;
}

// ==================================================================================
// 2. COMPUTE ENGINE
// ==================================================================================

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

// ==================================================================================
// 3. CONTROLLER: WS + SLIDING WINDOW
// ==================================================================================

void run_accelerator_optimized() {
    printf("--- SIMULATION: WEIGHT STATIONARY + INPUT SLIDING WINDOW ---\n");
    
    // Reset counters
    total_weight_bytes_loaded = 0;
    total_ifm_bytes_loaded = 0;
    
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;

    // Loop Pass (Weight Stationary)
    for (int p = 0; p < num_passes; p++) {
        printf("Pass %d/%d: Loading Weights...\n", p+1, num_passes);
        dma_load_weights(p);

        // Loop Height
        for (int ho = 0; ho < OUTPUT_H; ho++) {
            
            // --- PIXEL ĐẦU TIÊN CỦA HÀNG (wo=0) ---
            // Phải load đầy đủ (Warm-up buffer)
            dma_load_ifm_init(ho, p);
            
            // Tính toán
            int32_t res = run_pe_array();
            ofm_dram[ho * OUTPUT_W + 0] += res;

            // --- CÁC PIXEL CÒN LẠI (wo > 0) ---
            // Dùng kỹ thuật Sliding Window
            for (int wo = 1; wo < OUTPUT_W; wo++) {
                
                // Shift trái và load cột mới
                dma_shift_and_load_col(ho, wo, p);

                // Tính toán
                int32_t partial_result = run_pe_array();
                ofm_dram[ho * OUTPUT_W + wo] += partial_result;
            }
        }
    }

    printf("\n--- SIMULATION COMPLETED ---\n");
    printf("Total Weight Bytes Loaded: %" PRIu64 " bytes\n", total_weight_bytes_loaded);
    printf("Total IFM Bytes Loaded: %" PRIu64 " bytes\n", total_ifm_bytes_loaded);
}

void write_dram_to_file() {
    FILE* f = fopen("../ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

void cleanup() { free(ifm_dram); free(weight_dram); free(ofm_dram); }

int main() {
    dram_init();
    run_accelerator_optimized();
    write_dram_to_file();
    cleanup();
    return 0;
}