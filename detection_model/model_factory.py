from ultralytics import YOLO


def build_model(model_name):
    """Build and return a YOLO model based on the specified model name."""
    if model_name == "yolov8":
        # Load a YOLOv11 model
        model = YOLO('yolo8n.pt')
    elif model_name == "yolov11":
        # Load a YOLOv11 model
        model = YOLO('yolo11n.pt')
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model
