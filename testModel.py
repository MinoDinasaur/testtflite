import os
import psutil
import time
import csv
from ultralytics import YOLO

# Tên model (tflite hoặc pt đều dùng được nếu ultralytics hỗ trợ)
model_path = "v6/bestv6_saved_model/bestv6_full_integer_quant.tflite"
model = YOLO(model_path)

# Thư mục chứa ảnh
image_folder = "dataset/valid/images"
image_list = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Thông tin tiến trình để đo RAM
process = psutil.Process()

# Tên file CSV để ghi kết quả
model_name = os.path.splitext(os.path.basename(model_path))[0]
csv_file = f"inference_results_{model_name}.csv"

# Ghi header vào file CSV
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Inference Time (ms)", "RAM Used (MB)"])

    # Inference từng ảnh
    for img_path in image_list:
        mem_before = process.memory_info().rss / 1024 / 1024
        t0 = time.time()
        results = model(img_path)
        t1 = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024
        delta = mem_after - mem_before

        inference_time_ms = (t1 - t0) * 1000
        img_name = os.path.basename(img_path)

        print(f"[{img_name}] ΔRAM: {delta:.2f} MB | Time: {inference_time_ms:.2f} ms")

        # Ghi dòng dữ liệu vào CSV
        writer.writerow([img_name, f"{inference_time_ms:.2f}", f"{delta:.2f}"])
