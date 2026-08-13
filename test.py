from ultralytics import YOLO

try:
    model = YOLO("models/best_model.pt")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:")
    print(e)
