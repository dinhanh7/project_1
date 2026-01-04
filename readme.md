IFM: params/ifm.txt ,shape   : [1, 112, 112, 32]
WEIGHTS: params/weights.txt ,shape   : [1, 3, 3, 32]
OFM golden: golden_output/ofm_golden.txt ,shape   : [1, 112, 112, 1]
Stride=(1,1)
Padding = same

Viết lại vòng for trong phép Conv2D: có lệnh load, store, tính toán, share buffer.
Sắp xếp để tính toán sử dụng các phương pháp: unroll, tiling, tái sử dụng weight, input share.

Code bằng C -> đo latency, sử dụng bao nhiêu memory
