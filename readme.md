IFM: shape   : [1, 112, 112, 32]
WEIGHTS: shape   : [1, 3, 3, 32]
OFM: shape   : [1, 112, 112, 1]

Viết lại vòng for trong phép Conv2D: có lệnh load, store, tính toán, share buffer.
Sắp xếp để tính toán sử dụng các phương pháp: unroll, tailing, tái sử dụng weight, input share.

Code bằng C -> đo latency, sử dụng bao nhiêu memory