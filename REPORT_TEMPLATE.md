# CSC4005 Lab 6 Report – Export ONNX + Consistency Test + Benchmark

## 1. Thông tin

- Họ tên: Lưu Thanh Tùng
- Mã sinh viên: 1771040029
- Lớp: KHMT 1701

- Họ tên: Nguyễn Hoàng Anh
- Mã sinh viên: 1771040002
- Lớp: KHMT 1701

- Link GitHub repo: https://github.com/FIT-DNU-CS-16-01/fit-dnu-cs-16-01-17-01-csc4005-csc4005_lab6_onnx_starter_kit-csc4005_lab6_onnx_starter_kit
- Link checkpoint hoặc mô tả checkpoint sử dụng: checkpoints/best_model.pt (lấy từ lab ViT trước, dung lượng đo được ~327.37 MB)
- Link file ONNX nếu không commit trực tiếp: outputs/vit_smartcampus.onnx (dung lượng ~327.72 MB, lưu cục bộ do file lớn)

## 2. Mô tả mô hình đầu vào

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset |
| Số lớp | 5 |
| Classes | classroom, computerroom, library, corridor, office |
| Model PyTorch | Vision Transformer vit_b_16 |
| Checkpoint | checkpoints/best_model.pt (total_params=85,802,501; trainable_params=3,845) |
| Image size | 224x224 |
| Train mode từ lab trước | head_only |

## 3. Export ONNX

Điền thông tin:

| Thông số | Giá trị |
|---|---|
| ONNX path | outputs/vit_smartcampus.onnx |
| Opset | 17 |
| Dynamic batch | yes |
| Input name | input |
| Output name | logits |
| Model size | 327.72 MB |

Lệnh đã chạy:

```bash
conda activate HocSau
python -m src.export_onnx \
	--checkpoint checkpoints/best_model.pt \
	--onnx_path outputs/vit_smartcampus.onnx \
	--model_name vit_b_16 \
	--img_size 224 \
	--opset 17 \
	--dynamic_batch
```

Trạng thái export: exported_and_checked (đã qua ONNX checker).

## 4. Consistency Test

| Metric | Giá trị |
|---|---:|
| passed | true |
| num_samples | 32 |
| batch_size | 8 |
| max_abs_diff | 4.291534423828125e-05 |
| mean_abs_diff | 9.5015391252673e-06 |
| pred_match_rate | 1.0 |
| atol | 1e-4 |
| rtol | 1e-3 |

Lệnh đã chạy:

```bash
conda activate HocSau
python -m src.consistency_test \
	--checkpoint checkpoints/best_model.pt \
	--onnx_path outputs/vit_smartcampus.onnx \
	--num_samples 32 \
	--batch_size 8 \
	--atol 1e-4 \
	--rtol 1e-3 \
	--use_wandb \
	--wandb_project csc4005-lab6-onnx \
	--wandb_run_name consistency_onnx_lab6_full \
	--wandb_mode online
```

W&B run: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-onnx/runs/8osxvqzg

Nhận xét:

- PyTorch và ONNX có nhất quán không?
- Nếu có sai khác, sai khác lớn hay nhỏ?
- Sai khác này có làm thay đổi nhãn dự đoán không?

Trả lời:

- Có, hai runtime nhất quán tốt (passed=true).
- Sai khác logits rất nhỏ (max_abs_diff ~ 4.29e-05, mean_abs_diff ~ 9.50e-06), nằm trong ngưỡng atol/rtol đã đặt.
- Không làm thay đổi nhãn dự đoán vì pred_match_rate = 1.0.

## 5. Benchmark

Môi trường benchmark:

- Hệ điều hành: Windows 10 (10.0.19045)
- Python: 3.10.20
- CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
- Runtime so sánh: PyTorch vs ONNXRuntime (CPU)
- Cấu hình đo: batch_sizes = [1, 4, 8], warmup = 10, repeat = 50

| Runtime | Batch size | Mean latency (ms) | Median latency (ms) | P95 latency (ms) | Throughput (img/s) | Model size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 1 | 204.09 | 201.29 | 221.10 | 4.90 | 327.37 |
| ONNXRuntime | 1 | 171.54 | 169.95 | 195.55 | 5.83 | 327.72 |
| PyTorch | 4 | 709.85 | 699.92 | 790.72 | 5.63 | 327.37 |
| ONNXRuntime | 4 | 693.26 | 680.31 | 751.95 | 5.77 | 327.72 |
| PyTorch | 8 | 1443.04 | 1406.21 | 1650.12 | 5.54 | 327.37 |
| ONNXRuntime | 8 | 1470.48 | 1452.94 | 1651.12 | 5.44 | 327.72 |

Lệnh đã chạy:

```bash
conda activate HocSau
python -m src.benchmark \
	--checkpoint checkpoints/best_model.pt \
	--onnx_path outputs/vit_smartcampus.onnx \
	--batch_sizes 1 4 8 \
	--warmup 10 \
	--repeat 50 \
	--use_wandb \
	--wandb_project csc4005-lab6-onnx \
	--wandb_run_name benchmark_onnx_lab6_full_1050 \
	--wandb_mode online
```

W&B run: https://wandb.ai/thanhtung-contact-official-/csc4005-lab6-onnx/runs/we8dmiy3

## 6. Phân tích kết quả

Trả lời:

1. ONNXRuntime có nhanh hơn PyTorch không?
2. Batch size ảnh hưởng thế nào đến latency và throughput?
3. Vì sao cần warm-up trước khi đo benchmark?
4. Vì sao không nên chỉ đo một lần rồi kết luận?
5. Nếu triển khai lên CPU/edge device, bạn chọn batch size nào? Vì sao?

Phân tích:

1. ONNXRuntime nhanh hơn PyTorch ở batch size 1 và 4 (mean latency thấp hơn). Ở batch size 8, ONNXRuntime chậm hơn nhẹ.
2. Khi tăng batch size từ 1 lên 8, latency mỗi batch tăng mạnh ở cả hai runtime. Throughput trong thí nghiệm này tăng từ batch 1 lên 4 không đáng kể và giảm ở batch 8, cho thấy CPU đã gần bão hòa.
3. Warm-up giúp loại bỏ chi phí khởi tạo ban đầu (load kernel, cấp phát bộ nhớ, tối ưu đồ thị), làm số đo ổn định hơn.
4. Chỉ đo một lần dễ bị nhiễu bởi tiến trình nền, cache và scheduler của hệ điều hành; dùng repeat nhiều lần rồi lấy mean/median/p95 đáng tin cậy hơn.
5. Nếu triển khai trên CPU/edge cho bài toán gần thời gian thực, chọn batch size 1 với ONNXRuntime vì latency thấp nhất trong các cấu hình đo (171.54 ms) và throughput không thua kém.

## 7. Liên hệ triển khai thực tế

Viết 5–8 dòng:

- ONNX giúp gì trong triển khai mô hình?
- Consistency test giúp phát hiện lỗi gì?
- Benchmark giúp ra quyết định kỹ thuật như thế nào?
- Nếu cần đưa mô hình vào hệ thống Smart Campus thật, còn cần kiểm thử thêm điều gì?

ONNX tạo định dạng trung gian giúp mô hình dễ triển khai trên nhiều runtime và nền tảng khác nhau, không phụ thuộc chặt vào mã huấn luyện PyTorch.
Consistency test giúp phát hiện lỗi sai preprocessing, sai mapping tensor input/output, sai chế độ eval hoặc sai số xuất hiện sau export.
Benchmark cung cấp dữ liệu định lượng về latency, throughput và p95 để chọn runtime/batch size theo mục tiêu hệ thống.
Với bài toán Smart Campus, số liệu benchmark giúp cân bằng giữa tốc độ phản hồi và mức tải xử lý theo thời điểm.
Trước khi triển khai thực tế, cần kiểm thử thêm độ ổn định khi chạy dài hạn, tải đồng thời nhiều request, độ chính xác theo từng lớp trong dữ liệu thật và hành vi khi input nhiễu/thiếu sáng.
Ngoài ra có thể đánh giá thêm quantization hoặc tối ưu backend để giảm độ trễ trên thiết bị edge.

## 8. Kết luận

Tóm tắt:

- Export ONNX thành công hay chưa?
- Consistency test có pass không?
- Runtime nào nhanh hơn?
- Bài học chính rút ra từ lab này là gì?

- Export ONNX thành công và file đã được kiểm tra hợp lệ (status: exported_and_checked).
- Consistency test pass với độ lệch logits nhỏ và pred_match_rate = 1.0.
- ONNXRuntime nhanh hơn PyTorch ở batch size 1 và 4, còn batch size 8 thì PyTorch nhỉnh hơn nhẹ trong điều kiện CPU hiện tại.
- Bài học chính: quy trình triển khai cần đi theo chuỗi export -> kiểm thử nhất quán -> benchmark định lượng, không nên kết luận chỉ từ một phép đo đơn lẻ.
