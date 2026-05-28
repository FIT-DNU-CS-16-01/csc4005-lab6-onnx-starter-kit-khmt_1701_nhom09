# Hướng dẫn nộp bài bằng GitHub Classroom

## 1. Clone và làm bài

```bash
git clone <student-repo-url>
cd csc4005-lab6-onnx-starter-kit-khmt_1701_nhom09
```

## 2. Chạy pipeline Lab 6

Thực hiện đủ 3 bước:

1. Export ONNX
2. Consistency test
3. Benchmark

Artefact cần có trong thư mục outputs:

- export_report.json
- consistency_report.json
- benchmark_results.csv
- benchmark_summary.json

## 3. Commit và push

```bash
git add .
git commit -m "Complete Lab 6 ONNX export, consistency and benchmark"
git push origin main
```

## 4. Kiểm tra trước khi nộp

- Không push dataset/checkpoint/file lớn không cần thiết.
- README hoặc báo cáo có lệnh tái lập đầy đủ.
- Có kết quả và nhận xét theo rubric.
