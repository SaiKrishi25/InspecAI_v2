from ultralytics import YOLO

# Load YOLOv8 model (nano version)
model = YOLO("yolov8n.pt")

# Train with custom hyperparameters
model.train(
    data="data.yaml",    # dataset yaml
    epochs=50,           # training epochs
    imgsz=640,           # image size
    batch=16,            # batch size
    lr0=0.01,            # initial learning rate
    optimizer="AdamW",   # optimizer (SGD, Adam, AdamW)
    patience=20,         # early stopping patience
    weight_decay=0.0005  # regularization
)