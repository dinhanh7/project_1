#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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
// #define BUFFER_SIZE_BYTES 144   // 1152 bit = 144 bytes
// #define PARALLEL_CHANNELS 16    // 48 PE * 3 MACs / 9 weights = 16 channels
int NUM_PE, MACS_PER_PE, BUFFER_SIZE_BYTES;
int PARALLEL_CHANNELS;

// --- CẤU HÌNH HIỆU NĂNG ---
#define SYSTEM_FREQ_MHZ 100.0   
#define DRAM_BUS_WIDTH_BYTES 8  
#define PE_COMPUTE_CYCLES 1     

// Biến toàn cục đếm hiệu năng
unsigned long long total_dma_cycles = 0;
unsigned long long total_compute_cycles = 0;

// MÔ PHỎNG BỘ NHỚ (DRAM & BUFFERS)
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

// Hai Buffer riêng biệt theo yêu cầu
// int8_t buffer_ifm[BUFFER_SIZE_BYTES];   // Sẽ thay đổi liên tục (Sliding Window)
// int8_t buffer_weight[BUFFER_SIZE_BYTES]; // Sẽ ĐỨNG YÊN (Stationary) trong thời gian dài
int8_t* buffer_ifm;
int8_t* buffer_weight;

void dram_init() {
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C);
    // Load IFM
    FILE* f_ifm = fopen("params/ifm.txt", "r");
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
    // Weights
    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, 1);
    FILE* f_w = fopen("params/weights.txt", "r");
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

// CÁC HÀM DMA RIÊNG BIỆT (WEIGHT vs IFM)

// Hàm load Weight vào Buffer (1 lan moi pass)
void dma_load_weights(int pass_idx) {
    // Xác định channel bắt đầu cho pass hiện tại (ví dụ: pass 0 -> ch 0-15, pass 1 -> ch 16-31)
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    //load du so luong channel song song
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        //lay toan bo kernel 3x3 cho channel hien tai
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Lấy Weight từ DRAM
                int w_dram_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                 kw * (INPUT_C * OUTPUT_F) + 
                                 current_c * OUTPUT_F + 0;
                buffer_weight[buffer_ptr++] = weight_dram[w_dram_idx];
            }
        }
    }
    
    // Tính Latency: Load đầy 144 bytes weight
    // Overhead setup DMA + Transfer time
    int cycles = (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    total_dma_cycles += cycles;
}

// Hàm load IFM vào Buffer (Chạy liên tục cho từng pixel)
void dma_load_ifm(int ho, int wo, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Tính toán tọa độ trên Input dựa vào Output, Stride và Padding
                int hi = ho * STRIDE + kh - PADDING;
                int wi = wo * STRIDE + kw - PADDING;
                
                int8_t val = 0;
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                    val = ifm_dram[dram_idx];
                }
                buffer_ifm[buffer_ptr++] = val;
            }
        }
    }

    // Tính Latency: Load 144 bytes IFM
    int cycles = (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    total_dma_cycles += cycles;
}

// COMPUTE ENGINE

int32_t run_pe_array() {
    int32_t partial_sum = 0;
    
    // 48 PE chạy song song
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        
        for (int k = 0; k < MACS_PER_PE; k++) {
            // IFM lấy từ buffer IFM (mới load)
            // Weight lấy từ buffer Weight (đã load từ trước và giữ nguyên)
            int8_t a = buffer_ifm[base_idx + k];
            int8_t b = buffer_weight[base_idx + k];
            pe_acc += (int32_t)a * (int32_t)b;
        }
        partial_sum += pe_acc;
    }
    
    total_compute_cycles += PE_COMPUTE_CYCLES;
    return partial_sum;
}

// CONTROLLER: WEIGHT STATIONARY DATAFLOW

void run_accelerator_ws() {
    printf("--- STARTING WEIGHT STATIONARY SIMULATION ---\n");
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS; // de luon lam tron len

    // Đây là cốt lõi của Weight Stationary. Ta duyệt qua từng khối channel.
    for (int p = 0; p < num_passes; p++) {
        
        printf("Processing Pass %d/%d (Loading Weights to SRAM)...\n", p+1, num_passes);
        
        // Dữ liệu này sẽ nằm im trong buffer_weight cho đến khi tính xong 16 channel của ảnh
        dma_load_weights(p);

        // Quét toàn bộ 16 channel của ảnh với bộ Weight hiện tại
        for (int ho = 0; ho < OUTPUT_H; ho++) {
            for (int wo = 0; wo < OUTPUT_W; wo++) {
                
                // LOAD IFM (Liên tục load dữ liệu mới)
                dma_load_ifm(ho, wo, p);

                // COMPUTE
                int32_t partial_result = run_pe_array();

                // ACCUMULATE 
                // Vì ta tính theo từng Pass, nên ta phải cộng dồn vào kết quả cũ trong DRAM
                int out_idx = ho * OUTPUT_W + wo;
                ofm_dram[out_idx] += partial_result;
            }
        }
    }

    // Report
    unsigned long long total_cycles = total_dma_cycles + total_compute_cycles;
    double total_time_ms = (double)total_cycles / (SYSTEM_FREQ_MHZ * 1000.0);

    printf("\n--- PERFORMANCE REPORT (WEIGHT STATIONARY) ---\n");
    printf("Total Cycles: %llu\n", total_cycles);
    printf("  - DMA Cycles:     %llu\n", total_dma_cycles);
    printf("  - Compute Cycles: %llu\n", total_compute_cycles);
    printf("Estimated Time: %.4f ms\n", total_time_ms);
    printf("----------------------------------------------\n");
}

void write_dram_to_file() {
    FILE* f = fopen("ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

void cleanup() {
    free(ifm_dram); free(weight_dram); free(ofm_dram);
}

// int main() {
//     dram_init();
//     run_accelerator_ws(); // Chạy phiên bản Weight Stationary
//     write_dram_to_file();
//     cleanup();
//     return 0;
// }
int main(int argc, char *argv[]) {
    // Kiểm tra số lượng tham số (13 tham số + 1 tên file = 14)
    if (argc < 14) {
        printf("Usage: %s IH IW IC KH KW OF OH OW S P NPE MAC BUF\n", argv[0]);
        return -1;
    }

    // Gán giá trị từ Terminal vào biến
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

    // Tự động tính toán PARALLEL_CHANNELS
    // Logic: Tổng tài nguyên tính toán (PE * MACs) chia cho kích thước 1 kernel
    // Ví dụ: (48 * 3) / (3 * 3) = 144 / 9 = 16 channels
    int kernel_size = KERNEL_H * KERNEL_W;
    if (kernel_size > 0) {
        PARALLEL_CHANNELS = (NUM_PE * MACS_PER_PE) / kernel_size;
    } else {
        PARALLEL_CHANNELS = 1; // Giá trị mặc định an toàn
    }
    
    printf("--- Configuration ---\n");
    printf("Auto-calculated PARALLEL_CHANNELS: %d\n", PARALLEL_CHANNELS);
    printf("Buffer Size: %d bytes\n", BUFFER_SIZE_BYTES);

    // Cấp phát bộ nhớ động cho 2 Buffer
    buffer_ifm = (int8_t*)malloc(BUFFER_SIZE_BYTES * sizeof(int8_t));
    buffer_weight = (int8_t*)malloc(BUFFER_SIZE_BYTES * sizeof(int8_t));

    if (!buffer_ifm || !buffer_weight) {
        printf("Error: Memory allocation failed!\n");
        return -1;
    }

    // Chạy quy trình mô phỏng
    dram_init();          // Khởi tạo DRAM với kích thước mới
    run_accelerator_ws(); // Chạy mô phỏng Weight Stationary
    write_dram_to_file(); // Ghi kết quả
    
    // Dọn dẹp bộ nhớ
    free(buffer_ifm);
    free(buffer_weight);
    cleanup(); // Free DRAM

    return 0;
}