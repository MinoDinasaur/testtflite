from ultralytics import YOLO

model = YOLO('v8/bestv8.pt')  # Đường dẫn model .pt gốc

model.export(
    format='tflite',         # Xuất sang TensorFlow Lite
    int8=True,               # Bật int8 quantization
    imgsz=640,               # Kích thước ảnh đầu vào
    data='dataset/data.yaml'  # File data.yaml chứa đường dẫn ảnh calibration
)
