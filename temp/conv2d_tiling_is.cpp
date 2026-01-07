#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH BÀI TOÁN ---
// Kích thước Input Feature Map (IFM): 112x112, 32 channels
#define INPUT_H 112
#define INPUT_W 112
#define INPUT_C 32
// Kích thước Kernel (Bộ lọc): 3x3
#define KERNEL_H 3
#define KERNEL_W 3
// Số lượng Output Feature Map (OFM): 1 channel
#define OUTPUT_F 1 
#define OUTPUT_H 112
#define OUTPUT_W 112
#define STRIDE 1
#define PADDING 1

// --- CẤU HÌNH PHẦN CỨNG ---
// Số lượng phần tử xử lý (Processing Elements)
#define NUM_PE 48               
// Số phép nhân cộng (MAC) trên mỗi PE
#define MACS_PER_PE 3           
// Kích thước bộ nhớ đệm cục bộ (SRAM) trên chip: 144 bytes
// Tính toán: 16 channels song song * 9 weights (3x3) * 1 byte = 144 bytes
#define BUFFER_SIZE_BYTES 144   
// Số kênh xử lý song song trong một lần chạy (Pass)
#define PARALLEL_CHANNELS 16    

// --- CẤU HÌNH HIỆU NĂNG ---
#define SYSTEM_FREQ_MHZ 100.0   // Tần số hoạt động 100 MHz
#define DRAM_BUS_WIDTH_BYTES 8  // Độ rộng bus dữ liệu DRAM (64-bit)
#define PE_COMPUTE_CYCLES 1     // Số chu kỳ để PE thực hiện tính toán

// Biến toàn cục để đếm số chu kỳ (Cycles) cho việc đánh giá hiệu năng
unsigned long long total_dma_cycles = 0;     // Chu kỳ truy cập bộ nhớ (DMA)
unsigned long long total_compute_cycles = 0; // Chu kỳ tính toán

// --- MEMORY ---
// Mô phỏng bộ nhớ DRAM (Off-chip memory)
int8_t* ifm_dram;       // Dữ liệu đầu vào
int8_t* weight_dram;    // Trọng số
int32_t* ofm_dram;      // Dữ liệu đầu ra (kết quả)

// Mô phỏng bộ nhớ đệm (On-chip buffer/SRAM)
int8_t buffer_ifm[BUFFER_SIZE_BYTES];   
int8_t buffer_weight[BUFFER_SIZE_BYTES]; 

// Hàm khởi tạo dữ liệu giả lập trong DRAM
void dram_init() {
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C);
    // Init dữ liệu giả lập: tạo mẫu pattern đơn giản
    for(int i=0; i<INPUT_H*INPUT_W*INPUT_C; i++) ifm_dram[i] = (i % 100);

    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, 1);
    // Init trọng số bằng 1 để dễ kiểm tra tính toán
    for(int i=0; i<KERNEL_H*KERNEL_W*INPUT_C*OUTPUT_F; i++) weight_dram[i] = 1;

    ofm_dram = (int32_t*)calloc(OUTPUT_H * OUTPUT_W * OUTPUT_F, sizeof(int32_t));
}

// ==================================================================================
// 1. INPUT SLIDING WINDOW LOGIC (LOGIC CỬA SỔ TRƯỢT CHO INPUT)
// ==================================================================================

// [INIT] Load toàn bộ 3x3 block (Chỉ chạy tại cột đầu tiên wo=0)
// Hàm này tải dữ liệu "lạnh" từ DRAM vào Buffer khi bắt đầu một hàng mới.
// Nó phải tải đủ 9 phần tử (3x3) cho mỗi kênh đang xử lý song song.
void dma_load_ifm_full(int ho, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Tính toán tọa độ thực tế trên ảnh gốc (có tính padding)
                int hi = ho * STRIDE + kh - PADDING;
                int wi = 0 * STRIDE + kw - PADDING; // wo=0 vì đang ở đầu hàng
                
                int8_t val = 0;
                // Kiểm tra biên (Boundary check) để xử lý Padding = 0
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
                }
                buffer_ifm[buffer_ptr++] = val;
            }
        }
    }
    // Tính toán độ trễ: Full Load 144 bytes
    // Công thức: (Số bytes + Độ rộng Bus - 1) / Độ rộng Bus -> Làm tròn lên
    total_dma_cycles += (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
}

// [SLIDING] Shift trái buffer và chỉ load cột mới (Chạy tại các cột wo > 0)
// Đây là kỹ thuật tối ưu quan trọng: Tái sử dụng dữ liệu (Data Reuse).
// Thay vì tải lại 9 điểm ảnh, ta giữ lại 6 điểm cũ và chỉ tải 3 điểm mới.
void dma_shift_and_load_ifm(int ho, int wo, int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    
    // 1. SHIFT BUFFER (Mô phỏng dịch chuyển thanh ghi phần cứng)
    // Dữ liệu trong buffer được dời sang trái: Cột 1 -> Cột 0, Cột 2 -> Cột 1.
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int base = i * 9; // Mỗi kênh chiếm 9 vị trí trong buffer
        // Dời cột 1 về 0
        buffer_ifm[base + 0] = buffer_ifm[base + 1]; 
        buffer_ifm[base + 3] = buffer_ifm[base + 4]; 
        buffer_ifm[base + 6] = buffer_ifm[base + 7]; 
        // Dời cột 2 về 1
        buffer_ifm[base + 1] = buffer_ifm[base + 2]; 
        buffer_ifm[base + 4] = buffer_ifm[base + 5]; 
        buffer_ifm[base + 7] = buffer_ifm[base + 8]; 
    }

    // 2. LOAD NEW COLUMN (Chỉ tải cột thứ 3 mới nhất từ DRAM)
    int bytes_loaded = 0;
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;
        int base = i * 9;
        // Tính toán vị trí cột mới cần tải (cột index 2 trong window 3x3)
        int wi = wo * STRIDE + 2 - PADDING; 

        for (int kh = 0; kh < KERNEL_H; kh++) { 
            int hi = ho * STRIDE + kh - PADDING;
            int8_t val = 0;
            if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                val = ifm_dram[hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c];
            }
            // Ghi giá trị mới vào cột cuối cùng (index 2, 5, 8) của kênh đó trong buffer
            buffer_ifm[base + (kh * 3) + 2] = val; 
            bytes_loaded++;
        }
    }
    // Tính toán độ trễ: Partial Load 48 bytes (Nhanh gấp 3 lần so với full load)
    total_dma_cycles += (bytes_loaded + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
}

// ==================================================================================
// 2. WEIGHT LOADING (Mô phỏng Tiling: Load lại liên tục)
// ==================================================================================

// Hàm này sẽ được gọi TẠI MỖI PIXEL (WO) - Rất tốn kém băng thông
// Trong mô hình này, Weights KHÔNG được tái sử dụng (No Weight Reuse) theo chiều ngang (Output Stationary).
// Nó mô phỏng việc bộ nhớ đệm Weight quá nhỏ hoặc chiến lược điều khiển đơn giản.
void dma_load_weights_per_pixel(int pass_idx) {
    int channel_start = pass_idx * PARALLEL_CHANNELS;
    int buffer_ptr = 0;

    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break;

        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // Tính index phẳng của weight trong DRAM
                int w_idx = kh*(KERNEL_W*INPUT_C*OUTPUT_F) + kw*(INPUT_C*OUTPUT_F) + current_c*OUTPUT_F;
                buffer_weight[buffer_ptr++] = weight_dram[w_idx];
            }
        }
    }
    // Latency: Luôn load 144 bytes mỗi lần gọi -> Tốn nhiều chu kỳ DMA
    total_dma_cycles += (buffer_ptr + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
}

// ==================================================================================
// 3. COMPUTE ENGINE & CONTROLLER (BỘ TÍNH TOÁN VÀ ĐIỀU KHIỂN)
// ==================================================================================

// Mô phỏng mảng PE thực hiện phép nhân chập (Dot Product)
int32_t run_pe_array() {
    int32_t partial_sum = 0;
    // Duyệt qua các PE (Mỗi PE xử lý một phần của phép tính)
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        // Thực hiện MAC (Multiply-Accumulate)
        for (int k = 0; k < MACS_PER_PE; k++) {
            // Lấy dữ liệu từ buffer IFM và Weight để nhân
            pe_acc += (int32_t)buffer_ifm[base_idx + k] * (int32_t)buffer_weight[base_idx + k];
        }
        partial_sum += pe_acc;
    }
    // Cộng thêm thời gian tính toán vào tổng chu kỳ
    total_compute_cycles += PE_COMPUTE_CYCLES;
    return partial_sum;
}

// Hàm điều phối chính của mô phỏng
void run_simulation_hybrid() {
    printf("--- SIMULATION: TILING WEIGHTS + INPUT SLIDING WINDOW ---\n");
    // Tính số lần cần chạy (Pass) để xử lý hết tất cả các kênh Input (32 kênh / 16 kênh mỗi pass = 2 passes)
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;

    // Vòng lặp theo chiều cao Output
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        // Lưu ý: Đảo vòng lặp Pass ra ngoài Wo để giữ Buffer IFM cho Sliding Window
        // Nếu Pass nằm trong Wo, ta sẽ phải load lại IFM liên tục, mất tác dụng của Sliding Window.
        for (int p = 0; p < num_passes; p++) {
            
            // Vòng lặp theo chiều rộng Output
            for (int wo = 0; wo < OUTPUT_W; wo++) {
                
                // 1. WEIGHT LOADING (Kém hiệu quả - Theo yêu cầu bài toán)
                // Được gọi bên trong vòng lặp WO -> Load lại 112 lần mỗi hàng!
                // Đây là đặc điểm của chiến lược "Input Sliding Window" thuần túy mà không tối ưu Weight.
                dma_load_weights_per_pixel(p);

                // 2. IFM LOADING (Hiệu quả - Sliding Window)
                if (wo == 0) {
                    // Đầu hàng: Load đầy đủ
                    dma_load_ifm_full(ho, p); 
                } else {
                    // Giữa hàng: Tận dụng dữ liệu cũ, chỉ load phần mới
                    dma_shift_and_load_ifm(ho, wo, p); 
                }

                // 3. COMPUTE
                // Thực hiện tính toán trên dữ liệu đã có trong buffer
                int32_t res = run_pe_array();
                
                // Cộng dồn kết quả vào DRAM (Accumulation)
                // Vì ta chia nhỏ Channels thành nhiều Pass, nên kết quả của mỗi Pass là một phần của tổng (Partial Sum).
                ofm_dram[ho * OUTPUT_W + wo] += res;
            }
        }
    }

    // REPORT (BÁO CÁO KẾT QUẢ)
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

int main() {
    dram_init();
    run_simulation_hybrid();
    cleanup();
    return 0;
}