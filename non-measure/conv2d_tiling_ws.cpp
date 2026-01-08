#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#define PARALLEL_CHANNELS 16    // 48 PE * 3 MACs / 9 weights = 16 channels

// MÔ PHỎNG BỘ NHỚ (DRAM & BUFFERS)
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

// Hai Buffer riêng biệt theo yêu cầu
int8_t buffer_ifm[BUFFER_SIZE_BYTES];   // Sẽ thay đổi liên tục (Sliding Window)
int8_t buffer_weight[BUFFER_SIZE_BYTES]; // Sẽ ĐỨNG YÊN (Stationary) trong thời gian dài

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

    printf("--- WEIGHT STATIONARY SIMULATION COMPLETED ---\n");
}

void write_dram_to_file() {
    FILE* f = fopen("../ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

void cleanup() {
    free(ifm_dram); free(weight_dram); free(ofm_dram);
}

int main() {
    // Xóa file OFM cũ trước khi chạy
    remove("../ofm/ofm.txt");
    
    dram_init();
    run_accelerator_ws(); // Chạy phiên bản Weight Stationary
    write_dram_to_file();
    cleanup();
    return 0;
}