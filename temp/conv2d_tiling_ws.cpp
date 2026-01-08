#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// --- CẤU HÌNH BÀI TOÁN ---
// Định nghĩa kích thước Input Feature Map (IFM): 112x112, 32 channels
#define INPUT_H 112
#define INPUT_W 112
#define INPUT_C 32
// Kích thước Kernel (Weights): 3x3
#define KERNEL_H 3
#define KERNEL_W 3
// Số lượng Output Feature Map (OFM): 1 filter
#define OUTPUT_F 1 
// Kích thước Output: 112x112 (do padding=1, stride=1)
#define OUTPUT_H 112
#define OUTPUT_W 112
#define STRIDE 1
#define PADDING 1

// --- CẤU HÌNH PHẦN CỨNG ---
// Số lượng Processing Elements (PE): 48
#define NUM_PE 48               
// Mỗi PE thực hiện 3 phép nhân-cộng (MAC) mỗi chu kỳ
#define MACS_PER_PE 3           
// Kích thước bộ nhớ đệm trên chip (SRAM): 144 bytes
#define BUFFER_SIZE_BYTES 144   // 1152 bit = 144 bytes
// Số kênh xử lý song song trong 1 lần chạy: 16 channels
// (48 PE * 3 MACs) / (3x3 kernel size) = 144 / 9 = 16 channels
#define PARALLEL_CHANNELS 16    

// --- CẤU HÌNH HIỆU NĂNG ---
// Tần số hoạt động giả lập: 100 MHz
#define SYSTEM_FREQ_MHZ 100.0   
// Băng thông bộ nhớ DRAM: 8 bytes/cycle (64-bit bus)
#define DRAM_BUS_WIDTH_BYTES 8  
// Số cycle để PE tính toán xong 1 lượt
#define PE_COMPUTE_CYCLES 1     

// Biến toàn cục đếm hiệu năng (Cycles cho DMA và Compute)
unsigned long long total_dma_cycles = 0;
unsigned long long total_compute_cycles = 0;

// MÔ PHỎNG BỘ NHỚ (DRAM & BUFFERS)

// Con trỏ mô phỏng bộ nhớ ngoài (DRAM) chứa dữ liệu toàn cục
int8_t* ifm_dram;       
int8_t* weight_dram;    
int32_t* ofm_dram;      

// Hai Buffer riêng biệt trên chip (SRAM) theo kiến trúc Weight Stationary
// buffer_ifm: Chứa dữ liệu đầu vào, thay đổi liên tục theo cửa sổ trượt (Sliding Window)
int8_t buffer_ifm[BUFFER_SIZE_BYTES];   
// buffer_weight: Chứa trọng số, được giữ CỐ ĐỊNH (Stationary) trong thời gian dài để tái sử dụng
int8_t buffer_weight[BUFFER_SIZE_BYTES]; 

// Hàm khởi tạo DRAM: Cấp phát bộ nhớ và đọc dữ liệu từ file (hoặc tạo giả)
void dram_init() {
    // IFM: Cấp phát và đọc file params/ifm.txt
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C);
    FILE* f_ifm = fopen("params/ifm.txt", "r");
    if(f_ifm) {
        char line[64]; int idx = 0;
        while(fgets(line, 64, f_ifm)) ifm_dram[idx++] = (int8_t)atoi(line);
        fclose(f_ifm);
    } else { memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); } // Fallback nếu không có file

    // Weights: Cấp phát và đọc file params/weights.txt
    // Sắp xếp lại dữ liệu weight để phù hợp với việc truy xuất tuần tự
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
                             if (val > 0x7F) val -= 0x100; // Xử lý số âm 8-bit
                             // Tính index phẳng trong mảng 1 chiều
                             int idx = h*(KERNEL_W*INPUT_C*OUTPUT_F) + w*(INPUT_C*OUTPUT_F) + c*OUTPUT_F + f;
                             weight_dram[idx] = (int8_t)val;
                        }
        fclose(f_w);
    }

    // OFM: Cấp phát và reset về 0.
    // Quan trọng: Cần reset về 0 vì ta sẽ cộng dồn (accumulate) kết quả từ các pass khác nhau.
    ofm_dram = (int32_t*)calloc(OUTPUT_H * OUTPUT_W * OUTPUT_F, sizeof(int32_t));
}

// CÁC HÀM DMA RIÊNG BIỆT (WEIGHT vs IFM)

// Hàm load Weight vào Buffer (Đặc trưng của Weight Stationary: Chỉ chạy 1 lần mỗi Pass)
// Mục tiêu: Tối đa hóa việc tái sử dụng Weight đã load lên SRAM
void dma_load_weights(int pass_idx) {
    // Xác định channel bắt đầu cho pass hiện tại (ví dụ: pass 0 -> ch 0-15, pass 1 -> ch 16-31)
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    // Load đủ số lượng channel song song (PARALLEL_CHANNELS = 16)
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        // Load toàn bộ kernel 3x3 cho channel hiện tại
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
    
    // Tính Latency: Giả lập thời gian chuyển dữ liệu từ DRAM vào SRAM
    // Số cycles = (Tổng bytes cần truyền) / (Băng thông bytes/cycle)
    int cycles = (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    total_dma_cycles += cycles;
}

// Hàm load IFM vào Buffer (Chạy liên tục cho từng pixel đầu ra)
// Đây là phần tốn băng thông nhất vì dữ liệu IFM thay đổi liên tục khi cửa sổ trượt
void dma_load_ifm(int ho, int wo, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    // Load vùng dữ liệu IFM tương ứng với vị trí cửa sổ trượt (ho, wo)
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Tính toán tọa độ trên Input dựa vào Output, Stride và Padding
                int hi = ho * STRIDE + kh - PADDING;
                int wi = wo * STRIDE + kw - PADDING;
                
                int8_t val = 0;
                // Kiểm tra biên (Boundary check) để xử lý Padding zero
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                    val = ifm_dram[dram_idx];
                }
                buffer_ifm[buffer_ptr++] = val;
            }
        }
    }

    // Tính Latency cho việc load IFM
    int cycles = (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    total_dma_cycles += cycles;
}

// COMPUTE ENGINE

// Hàm mô phỏng mảng tính toán (PE Array)
// Thực hiện nhân chập giữa dữ liệu trong buffer_ifm và buffer_weight
int32_t run_pe_array() {
    int32_t partial_sum = 0;
    
    // 48 PE chạy song song, mỗi PE xử lý một phần của phép tính
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        
        // Mỗi PE thực hiện 3 phép MAC (Multiply-Accumulate)
        for (int k = 0; k < MACS_PER_PE; k++) {
            // IFM lấy từ buffer IFM (mới load)
            // Weight lấy từ buffer Weight (đã load từ trước và giữ nguyên - Stationary)
            int8_t a = buffer_ifm[base_idx + k];
            int8_t b = buffer_weight[base_idx + k];
            pe_acc += (int32_t)a * (int32_t)b;
        }
        partial_sum += pe_acc;
    }
    
    // Cộng thêm thời gian tính toán của PE vào tổng thời gian
    total_compute_cycles += PE_COMPUTE_CYCLES;
    return partial_sum;
}

// CONTROLLER: WEIGHT STATIONARY DATAFLOW

// Hàm điều khiển chính mô phỏng luồng dữ liệu Weight Stationary
void run_accelerator_ws() {
    printf("--- STARTING WEIGHT STATIONARY SIMULATION ---\n");
    // Chia nhỏ số kênh Input (32) thành các Pass (mỗi pass xử lý 16 kênh)
    // Do phần cứng giới hạn chỉ xử lý được 16 kênh song song
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS; // 32/16 = 2 passes

    // === OUTER LOOP: PASS (Channels) ===
    // Đây là cốt lõi của Weight Stationary: Vòng lặp Channel nằm ngoài cùng.
    // Ta giữ bộ Weight của 16 kênh này trên chip và dùng nó để quét toàn bộ ảnh.
    for (int p = 0; p < num_passes; p++) {
        
        printf("Processing Pass %d/%d (Loading Weights to SRAM)...\n", p+1, num_passes);
        
        // LOAD WEIGHT (Chỉ 1 lần cho cả Pass này)
        // Dữ liệu này sẽ nằm im trong buffer_weight cho đến khi tính xong hết ảnh (112x112 pixels)
        // Đây là lợi điểm: Giảm thiểu việc đọc lại Weight từ DRAM.
        dma_load_weights(p);

        // === INNER LOOPS: SPATIAL (Height/Width) ===
        // Quét toàn bộ ảnh không gian với bộ Weight hiện tại đang "đứng yên" trên chip
        for (int ho = 0; ho < OUTPUT_H; ho++) {
            for (int wo = 0; wo < OUTPUT_W; wo++) {
                
                // LOAD IFM (Liên tục load dữ liệu mới)
                // Dữ liệu IFM thay đổi theo từng pixel (ho, wo)
                dma_load_ifm(ho, wo, p);

                // COMPUTE
                // Tính toán tích chập tại điểm (ho, wo) cho nhóm kênh hiện tại
                int32_t partial_result = run_pe_array();

                // ACCUMULATE (Read-Modify-Write to DRAM/Output Buffer)
                // Vì ta tính theo từng Pass (Partial Output - Tổng riêng), 
                // nên ta phải cộng dồn kết quả mới vào kết quả cũ đã có trong DRAM.
                // Pass 1: Tính 16 kênh đầu -> Lưu vào DRAM.
                // Pass 2: Tính 16 kênh sau -> Cộng dồn vào DRAM -> Ra kết quả cuối cùng.
                int out_idx = ho * OUTPUT_W + wo;
                ofm_dram[out_idx] += partial_result;
            }
        }
    }

    // Report: Tính toán và báo cáo hiệu năng
    unsigned long long total_cycles = total_dma_cycles + total_compute_cycles;
    double total_time_ms = (double)total_cycles / (SYSTEM_FREQ_MHZ * 1000.0);

    printf("\n--- PERFORMANCE REPORT (WEIGHT STATIONARY) ---\n");
    printf("Total Cycles: %llu\n", total_cycles);
    printf("  - DMA Cycles:     %llu\n", total_dma_cycles);
    printf("  - Compute Cycles: %llu\n", total_compute_cycles);
    printf("Estimated Time: %.4f ms\n", total_time_ms);
    printf("----------------------------------------------\n");
}

// Ghi kết quả OFM từ DRAM ra file để kiểm tra
void write_dram_to_file() {
    FILE* f = fopen("ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

void cleanup() {
    free(ifm_dram); free(weight_dram); free(ofm_dram);
}

int main() {
    dram_init();
    run_accelerator_ws(); // Chạy phiên bản Weight Stationary
    write_dram_to_file();
    cleanup();
    return 0;
}