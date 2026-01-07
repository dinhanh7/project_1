#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// --- CẤU HÌNH BÀI TOÁN (PROBLEM CONFIG) ---
// Định nghĩa kích thước của Input Feature Map (IFM) và Kernel
#define INPUT_H 112      // Chiều cao ảnh đầu vào
#define INPUT_W 112      // Chiều rộng ảnh đầu vào
#define INPUT_C 32       // Số kênh (channels) đầu vào (độ sâu)
#define KERNEL_H 3       // Chiều cao bộ lọc (filter/kernel)
#define KERNEL_W 3       // Chiều rộng bộ lọc
#define OUTPUT_F 1       // Số lượng bộ lọc (số kênh đầu ra - Output Feature Map)
#define OUTPUT_H 112     // Chiều cao ảnh đầu ra (giả sử giữ nguyên kích thước)
#define OUTPUT_W 112     // Chiều rộng ảnh đầu ra
#define STRIDE 1         // Bước nhảy của cửa sổ trượt
#define PADDING 1        // Lề thêm vào xung quanh ảnh (để giữ kích thước output)

// --- CẤU HÌNH PHẦN CỨNG (HW SPEC) ---
// Định nghĩa tài nguyên phần cứng của Accelerator
#define NUM_PE 48               // Tổng số đơn vị xử lý (Processing Elements) có trong chip
#define MACS_PER_PE 3           // Số phép nhân-cộng (MAC) mà 1 PE có thể làm trong 1 chu kỳ (ví dụ: xử lý 3 phần tử của kernel 3x3 1 hàng)
#define BUFFER_SIZE_BYTES 144   // Kích thước bộ nhớ đệm trên chip (SRAM) tính bằng byte (đủ chứa dữ liệu cho 1 lần tính toán của các PE)
#define PARALLEL_CHANNELS 16    // Số kênh đầu vào (Input Channels) mà phần cứng có thể xử lý song song cùng lúc.
                                // Vì INPUT_C = 32 mà PARALLEL_CHANNELS = 16, ta sẽ cần chia làm 2 lần chạy (passes).

// --- CẤU HÌNH HIỆU NĂNG (PERFORMANCE METRICS) ---
// Các thông số để ước tính thời gian chạy
#define SYSTEM_FREQ_MHZ 100.0   // Tần số hoạt động của chip: 100 MHz
#define DRAM_BUS_WIDTH_BYTES 8  // Độ rộng bus dữ liệu nối với RAM ngoài: 64-bit = 8 bytes (mỗi chu kỳ truyền được 8 bytes)
#define PE_COMPUTE_CYCLES 1     // Số cycle để mảng PE hoàn thành 1 lượt tính toán (giả sử kiến trúc Pipelined lý tưởng)

// Biến toàn cục để lưu thống kê hiệu năng
unsigned long long total_dma_cycles = 0;     // Tổng số cycle tiêu tốn cho việc chuyển dữ liệu (DMA)
unsigned long long total_compute_cycles = 0; // Tổng số cycle tiêu tốn cho việc tính toán (Compute)
unsigned long long total_cycles = 0;         // Tổng thời gian toàn hệ thống

// ==================================================================================
// 1. MÔ PHỎNG DRAM (BỘ NHỚ NGOÀI)
// ==================================================================================
// Các con trỏ này đại diện cho bộ nhớ RAM vật lý chứa dữ liệu
int8_t* ifm_dram;       // Chứa dữ liệu đầu vào (Input Feature Map)
int8_t* weight_dram;    // Chứa trọng số (Weights/Kernels)
int32_t* ofm_dram;      // Chứa kết quả đầu ra (Output Feature Map) - dùng int32 để tránh tràn số khi cộng dồn

// Hàm khởi tạo và nạp dữ liệu vào "RAM"
void dram_init() {
    // Cấp phát bộ nhớ cho IFM
    ifm_dram = (int8_t*)malloc(INPUT_H * INPUT_W * INPUT_C * sizeof(int8_t));
    
    // Thử đọc dữ liệu IFM từ file (mô phỏng việc load ảnh/data thực tế)
    FILE* f_ifm = fopen("params/ifm.txt", "r");
    if(f_ifm) {
        char line[64];
        int idx = 0;
        while(fgets(line, 64, f_ifm)) ifm_dram[idx++] = (int8_t)atoi(line);
        fclose(f_ifm);
    } else {
        // Fallback: Nếu không có file, điền toàn bộ giá trị 1 để test luồng chạy
        memset(ifm_dram, 1, INPUT_H * INPUT_W * INPUT_C); 
    }

    // Cấp phát bộ nhớ cho Weights
    weight_dram = (int8_t*)calloc(KERNEL_H * KERNEL_W * INPUT_C * OUTPUT_F, sizeof(int8_t));
    
    // Thử đọc dữ liệu Weights từ file
    FILE* f_w = fopen("params/weights.txt", "r");
    if(f_w) {
        char line[64];
        // Vòng lặp lồng nhau để đọc weight theo thứ tự (F, H, W, C) hoặc tương tự tùy format
        for(int f=0; f<OUTPUT_F; f++)
            for(int h=0; h<KERNEL_H; h++)
                for(int w=0; w<KERNEL_W; w++)
                    for(int c=0; c<INPUT_C; c++)
                        if(fgets(line, 64, f_w)) {
                             int val = atoi(line);
                             // Xử lý số âm cho int8 (nếu file lưu dạng unsigned 0-255)
                             if (val > 0x7F) val -= 0x100;
                             // Tính chỉ số phẳng (flatten index) để lưu vào mảng 1 chiều
                             int idx = h*(KERNEL_W*INPUT_C*OUTPUT_F) + w*(INPUT_C*OUTPUT_F) + c*OUTPUT_F + f;
                             weight_dram[idx] = (int8_t)val;
                        }
        fclose(f_w);
    }

    // Cấp phát bộ nhớ cho kết quả đầu ra
    ofm_dram = (int32_t*)malloc(OUTPUT_H * OUTPUT_W * OUTPUT_F * sizeof(int32_t));
}

// ==================================================================================
// 2. MÔ PHỎNG BUFFER & DMA (BỘ NHỚ ĐỆM & TRUYỀN DỮ LIỆU)
// ==================================================================================
// Đây là bộ nhớ SRAM bên trong chip (On-chip memory), tốc độ rất nhanh nhưng dung lượng nhỏ
int8_t buffer_ifm[BUFFER_SIZE_BYTES];
int8_t buffer_weight[BUFFER_SIZE_BYTES];

// Hàm này mô phỏng hoạt động của DMA Controller:
// Nhiệm vụ: Copy dữ liệu cần thiết cho 1 lần tính toán từ DRAM vào Buffer.
// Input: Tọa độ đầu ra (ho, wo) và chỉ số pass (lần chạy thứ mấy cho channel).
// Output: Số cycle tiêu tốn để load dữ liệu.
int dma_load_buffers(int ho, int wo, int pass_idx) {
    // Xóa sạch buffer trước khi load mới
    memset(buffer_ifm, 0, BUFFER_SIZE_BYTES);
    memset(buffer_weight, 0, BUFFER_SIZE_BYTES);

    // Xác định dải channel cần load trong pass này (Tiling theo chiều Channel)
    int channel_start = pass_idx * PARALLEL_CHANNELS; 
    int buffer_ptr = 0; 
    int bytes_transferred = 0; // Biến đếm tổng số byte thực tế cần truyền qua bus

    // Vòng lặp load dữ liệu cho các channel song song
    for (int i = 0; i < PARALLEL_CHANNELS; i++) {
        int current_c = channel_start + i;
        if (current_c >= INPUT_C) break; // Đã hết channel thực tế

        // Quét qua cửa sổ kernel (3x3)
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                // --- 1. Tính toán địa chỉ IFM ---
                // Từ tọa độ output (ho, wo) suy ngược ra tọa độ input (hi, wi)
                // Công thức: Input = Output * Stride + Kernel_Offset - Padding
                int hi = ho * STRIDE + kh - PADDING;
                int wi = wo * STRIDE + kw - PADDING;
                
                int8_t val_ifm = 0;
                // Kiểm tra biên (Boundary Check) để xử lý Padding (nếu ra ngoài ảnh thì giá trị là 0)
                if (hi >= 0 && hi < INPUT_H && wi >= 0 && wi < INPUT_W) {
                    int dram_idx = hi * (INPUT_W * INPUT_C) + wi * INPUT_C + current_c;
                    val_ifm = ifm_dram[dram_idx]; // Đọc từ DRAM
                }

                // --- 2. Tính toán địa chỉ Weight ---
                int w_dram_idx = kh * (KERNEL_W * INPUT_C * OUTPUT_F) + 
                                 kw * (INPUT_C * OUTPUT_F) + 
                                 current_c * OUTPUT_F + 0; // Giả sử output channel 0
                int8_t val_w = weight_dram[w_dram_idx]; // Đọc từ DRAM

                // --- 3. Ghi vào Buffer ---
                buffer_ifm[buffer_ptr] = val_ifm;
                buffer_weight[buffer_ptr] = val_w;
                buffer_ptr++;
                
                // Giả lập: Mỗi lần tính toán cần load cả IFM và Weight mới
                // (Trong thực tế có thể tối ưu bằng Stationary dataflow để giảm load weight)
                bytes_transferred += 2; // 1 byte IFM + 1 byte Weight
            }
        }
    }

    // --- TÍNH TOÁN LATENCY (ĐỘ TRỄ) ---
    // Tổng số bytes cần truyền
    int total_bytes = buffer_ptr * 2; 
    
    // Thời gian truyền = Tổng bytes / Băng thông (Bytes/cycle)
    // Phép tính (A + B - 1) / B là mẹo để làm tròn lên (ceiling) trong số nguyên
    int cycles = (total_bytes + DRAM_BUS_WIDTH_BYTES - 1) / DRAM_BUS_WIDTH_BYTES;
    
    return cycles;
}

// ==================================================================================
// 3. MÔ PHỎNG COMPUTE ENGINE (KHỐI TÍNH TOÁN)
// ==================================================================================
// Hàm này mô phỏng mảng PE thực hiện phép nhân chập trên dữ liệu đã có trong Buffer
// Output: Kết quả tính toán (Partial Sum) và trả về số cycle tiêu tốn qua con trỏ cycles_taken
int32_t run_pe_array(int* cycles_taken) {
    int32_t partial_sum = 0;

    // --- Logic tính toán chức năng (Functional Simulation) ---
    // Duyệt qua các PE (mỗi PE chịu trách nhiệm một phần công việc)
    for (int pe_id = 0; pe_id < NUM_PE; pe_id++) {
        int base_idx = pe_id * MACS_PER_PE; 
        int32_t pe_acc = 0; 
        // Mỗi PE thực hiện MACS_PER_PE phép nhân cộng
        for (int k = 0; k < MACS_PER_PE; k++) { 
            // Lấy dữ liệu từ buffer (SRAM) - truy cập cực nhanh
            int8_t a = buffer_ifm[base_idx + k];
            int8_t b = buffer_weight[base_idx + k];
            pe_acc += (int32_t)a * (int32_t)b; // MAC: Multiply-Accumulate
        }
        partial_sum += pe_acc; // Cộng dồn kết quả từ các PE
    }

    // --- TÍNH TOÁN LATENCY ---
    // Vì các PE chạy song song (Parallel), thời gian thực thi chỉ tính bằng thời gian của 1 PE
    // (hoặc pipeline depth). Ở đây giả định là hằng số PE_COMPUTE_CYCLES.
    *cycles_taken = PE_COMPUTE_CYCLES;

    return partial_sum;
}

// ==================================================================================
// 4. CONTROLLER & REPORT (BỘ ĐIỀU KHIỂN & BÁO CÁO)
// ==================================================================================
// Hàm chính điều phối toàn bộ hoạt động của Accelerator
void run_accelerator() {
    printf("--- STARTING SIMULATION ---\n");
    printf("Specs:\n");
    printf("  - Frequency: %.1f MHz\n", SYSTEM_FREQ_MHZ);
    printf("  - DMA Bandwidth: %d Bytes/cycle\n", DRAM_BUS_WIDTH_BYTES);
    printf("  - PE Array: %d PEs (Parallel)\n", NUM_PE);
    printf("---------------------------\n");

    // Tính số lượng "pass" cần thiết để xử lý hết độ sâu (Channel) của input
    // Ví dụ: 32 kênh input, phần cứng xử lý được 16 kênh/lần -> cần 2 passes.
    int num_passes = (INPUT_C + PARALLEL_CHANNELS - 1) / PARALLEL_CHANNELS;
    
    // Reset các bộ đếm thống kê
    total_dma_cycles = 0;
    total_compute_cycles = 0;

    // --- Main Loop (Vòng lặp chính quét qua ảnh đầu ra) ---
    for (int ho = 0; ho < OUTPUT_H; ho++) {
        for (int wo = 0; wo < OUTPUT_W; wo++) {
            
            int32_t final_accumulator = 0; // Biến chứa giá trị pixel đầu ra tại (ho, wo)

            // Vòng lặp qua các pass (Tiling theo chiều sâu)
            for (int p = 0; p < num_passes; p++) {
                
                // BƯỚC 1: DMA Load - Chuyển dữ liệu từ DRAM vào Buffer
                // (CPU chờ DMA làm xong mới chạy tiếp - mô hình tuần tự đơn giản)
                int dma_c = dma_load_buffers(ho, wo, p);
                total_dma_cycles += dma_c;

                // BƯỚC 2: Compute - PE Array tính toán trên dữ liệu trong Buffer
                int comp_c = 0;
                int32_t pass_result = run_pe_array(&comp_c);
                total_compute_cycles += comp_c;
                
                // Cộng dồn kết quả từng pass (Partial Sum Accumulation)
                final_accumulator += pass_result;
            }

            // Ghi kết quả cuối cùng ra DRAM (Output Feature Map)
            int out_idx = ho * OUTPUT_W + wo; 
            ofm_dram[out_idx] = final_accumulator;
        }
    }
    
    // Tổng cycle = Thời gian DMA + Thời gian Tính toán
    // (Lưu ý: Trong kiến trúc hiện đại, DMA và Compute có thể chạy song song (Double Buffering) để ẩn độ trễ,
    // nhưng ở đây đang mô phỏng chạy tuần tự).
    total_cycles = total_dma_cycles + total_compute_cycles;

    // --- REPORT KẾT QUẢ ---
    // Tính thời gian thực tế (ms) = Số cycle / Tần số
    double total_time_ms = (double)total_cycles / (SYSTEM_FREQ_MHZ * 1000.0);
    
    printf("\n--- PERFORMANCE REPORT ---\n");
    printf("Total Output Pixels: %d\n", OUTPUT_H * OUTPUT_W);
    printf("Total Cycles: %llu\n", total_cycles);
    printf("  - DMA Cycles:     %llu (%.2f%%)\n", total_dma_cycles, (double)total_dma_cycles/total_cycles*100.0);
    printf("  - Compute Cycles: %llu (%.2f%%)\n", total_compute_cycles, (double)total_compute_cycles/total_cycles*100.0);
    printf("Estimated Time: %.4f ms\n", total_time_ms);
    printf("--------------------------\n");
}

// Ghi kết quả từ DRAM ảo ra file text để kiểm tra
void write_dram_to_file() {
    FILE* f = fopen("ofm/ofm.txt", "w");
    if (!f) return;
    for(int i=0; i<OUTPUT_H*OUTPUT_W; i++) fprintf(f, "%d\n", ofm_dram[i]);
    fclose(f);
}

// Giải phóng bộ nhớ
void cleanup() {
    free(ifm_dram);
    free(weight_dram);
    free(ofm_dram);
}

int main() {
    dram_init();        // 1. Khởi tạo bộ nhớ
    run_accelerator();  // 2. Chạy mô phỏng phần cứng
    write_dram_to_file(); // 3. Xuất kết quả
    cleanup();          // 4. Dọn dẹp
    return 0;
}