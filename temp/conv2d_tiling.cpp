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

// --- CẤU HÌNH PHẦN CỨNG (HW SPEC) ---
#define NUM_PE 48               // Tổng số PE
#define MACS_PER_PE 3           // Số phép MAC mỗi PE làm được trong 1 chu kỳ
#define BUFFER_SIZE_BYTES 144   // 1152 bits = 144 bytes (Đủ chứa dữ liệu cho 48 PE * 3 input)

// Input Channel xử lý song song = 48 PE / (9 weights / 3 MACs) = 16 Channels
#define PARALLEL_CHANNELS 16    

// MÔ PHỎNG DRAM (Data nằm trong bộ nhớ lớn)

int8_t* ifm_dram;       // DRAM chứa Input Feature Map
int8_t* weight_dram;    // DRAM chứa Weights (Lưu ý: Bạn yêu cầu Weight là int8)
int32_t* ofm_dram;      // DRAM chứa Output Feature Map (Kết quả)

// Hàm khởi tạo và đọc dữ liệu vào DRAM (Giả lập việc load model)
void dram_init() {
    // Cấp phát DRAM cho IFM
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C * sizeof(int8_t));
    FILE* f_ifm = fopen("../params/ifm.txt", "r");
    if(f_ifm) {
        char line[64];
        int idx = 0;
        while(fgets(line, 64, f_ifm)) ifm_dram[idx++] = (int8_t)atoi(line);
        fclose(f_ifm);
    } else { printf("Error loading IFM to DRAM\n"); exit(1); }

    // Cấp phát DRAM cho Weights (INT8)
    // Lưu ý: File weight gốc có thể là int16 hoặc int8, ở đây ta giả lập ép về int8 theo yêu cầu HW
    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, sizeof(int8_t));
    FILE* f_w = fopen("../params/weights.txt", "r");
    if(f_w) {
        char line[64];
        // Đọc theo thứ tự F->H->W->C nhưng map vào DRAM linear
        // Để đơn giản cho simulation, ta load linear và sẽ tính toán index lại khi fetch
        // Logic đọc phức tạp (F,H,W,C) đã được xử lý ở các bài trước, 
        // ở đây ta giả định DRAM lưu trữ tuyến tính theo F-H-W-C để dễ truy xuất block.
        // Tuy nhiên code cũ lưu H-W-C-F. Ta sẽ tái sử dụng logic H-W-C-F cho đồng bộ.
        for(int f=0; f<OUTPUT_F; f++)
            for(int h=0; h<KERNEL_H; h++)
                for(int w=0; w<KERNEL_W; w++)
                    for(int c=0; c<INPUT_C; c++)
                        if(fgets(line, 64, f_w)) {
                             int val = atoi(line);
                             if (val > 0x7F) val -= 0x100;
                             // Map H-W-C-F
                             int idx = h*(KERNEL_W*INPUT_C*OUTPUT_F) + w*(INPUT_C*OUTPUT_F) + c*OUTPUT_F + f;
                             weight_dram[idx] = (int8_t)val;
                        }
        fclose(f_w);
    }
    
    // Cấp phát DRAM cho OFM
    ofm_dram = (int32_t*)malloc(OUTPUT_H * OUTPUT_W * OUTPUT_F * sizeof(int32_t));
}

// MÔ PHỎNG ON-CHIP BUFFER (SRAM 1152 bit)

// Buffer phần cứng: 144 bytes
int8_t buffer_ifm[BUFFER_SIZE_BYTES];
int8_t buffer_weight[BUFFER_SIZE_BYTES];

// Hàm giả lập bộ điều khiển DMA (Direct Memory Access)
// Nhiệm vụ: Lôi dữ liệu từ DRAM ném vào Buffer
// pass_idx: Vì 32 channel không xử lý hết 1 lần, ta chia thành các pass (0 hoặc 1)
void dma_load_buffers(int ho, int wo, int pass_idx) {
    // Reset buffer (tùy chọn, để debug)
    memset(buffer_ifm, 0, BUFFER_SIZE_BYTES);
    memset(buffer_weight, 0, BUFFER_SIZE_BYTES);

    int channel_start = pass_idx * PARALLEL_CHANNELS; // Pass 0 -> ch 0, Pass 1 -> ch 16
    int buffer_ptr = 0; // Con trỏ ghi vào buffer (0 -> 143)

    // Duyệt qua 16 channel cần xử lý song song
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break; // Safety check

        // Với mỗi channel, load 3x3 kernel (9 phần tử)
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                
                // Fetch IFM từ DRAM (Xử lý Padding tại đây)
                int hi = ho * STRIDE + kh - PADDING;
                int wi = wo * STRIDE + kw - PADDING;
                
                int8_t val_ifm = 0;
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                    val_ifm = ifm_dram[dram_idx];
                }

                // Fetch Weight từ DRAM
                // Layout H-W-C-F
                int w_dram_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                 kw * (INPUT_C * OUTPUT_F) + 
                                 current_c * OUTPUT_F + 
                                 0; // Filter 0
                int8_t val_w = weight_dram[w_dram_idx];

                // Ghi vào Buffer
                buffer_ifm[buffer_ptr] = val_ifm;
                buffer_weight[buffer_ptr] = val_w;
                buffer_ptr++;
            }
        }
    }
    // Kết thúc hàm này, buffer đã đầy 144 bytes dữ liệu
}

// MÔ PHỎNG COMPUTE ENGINE (48 PEs)

// Mỗi PE thực hiện 3 MACs.
// Logic:
// - Hệ thống có 48 PE.
// - Dữ liệu trong buffer là 144 bytes.
// - Mỗi PE sẽ lấy 3 bytes từ Buffer IFM và 3 bytes từ Buffer Weight để nhân cộng.
// - Tổng hợp kết quả của 48 PE lại ("Reduction tree" hoặc "Adder tree").

int32_t run_pe_array() {
    int32_t partial_sum_cycle = 0;

    // Mô phỏng chạy song song 48 PE
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        
        // Xác định địa chỉ dữ liệu trong buffer mà PE này chịu trách nhiệm
        // PE 0: index 0,1,2
        // PE 1: index 3,4,5
        // ...
        int base_idx = pe_id * MACS_PER_PE; 

        int32_t pe_acc = 0; // Thanh ghi accumulator bên trong PE

        // PE thực hiện 3 phép MAC
        for (int k = 0; k < MACS_PER_PE; k++) {
            int8_t a = buffer_ifm[base_idx + k];
            int8_t b = buffer_weight[base_idx + k];
            pe_acc += (int32_t)a * (int32_t)b;
        }

        // Sau khi PE tính xong, kết quả được gom lại (Adder Tree)
        partial_sum_cycle += pe_acc;
    }

    return partial_sum_cycle;
}

// HỆ THỐNG ĐIỀU KHIỂN CHÍNH (CONTROLLER)

void run_accelerator() {
    printf("Starting HW Simulation...\n");
    printf("Config: %d PEs, Buffer %d bytes, Parallel Channels: %d\n", NUM_PE, BUFFER_SIZE_BYTES, PARALLEL_CHANNELS);

    // Tính toán số pass cần thiết (32 channels / 16 = 2 passes)
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;

    // Loop qua từng pixel đầu ra (H x W)
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        for (int wo = 0; wo < OUTPUT_W; wo++) {
            
            int32_t final_accumulator = 0; // Thanh ghi tích lũy global (sau các pass)

            // Loop các Pass (Time Multiplexing)
            for (int p = 0; p < num_passes; p++) {
                
                // Controller ra lệnh DMA load dữ liệu vào SRAM
                dma_load_buffers(ho, wo, p);

                // Controller kích hoạt PE Array chạy
                int32_t pass_result = run_pe_array();

                // Cộng dồn kết quả từng pass
                final_accumulator += pass_result;
            }

            // Write Back to DRAM (Store OFM)
            int out_idx = ho * OUTPUT_W + wo; // Filter=0
            ofm_dram[out_idx] = final_accumulator;
        }
    }
    printf("Simulation Complete.\n");
}

void write_dram_to_file() {
    FILE* f = fopen("../ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) {
        fprintf(f, "%d\n", ofm_dram[i]);
    }
    fclose(f);
}

// Dọn dẹp
void cleanup() {
    free(ifm_dram);
    free(weight_dram);
    free(ofm_dram);
}

int main() {
    // Khởi tạo hệ thống (Load DRAM)
    dram_init();

    // Chạy mô phỏng
    run_accelerator();

    // Xuất kết quả
    write_dram_to_file();

    // Kết thúc
    cleanup();
    return 0;
}