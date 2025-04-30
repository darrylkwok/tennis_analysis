from ultralytics import YOLO


# model = YOLO('training/runs/detect/train/weights/last.pt')

model = YOLO('yolov8x')

model.track('inputs/input_video.mp4', conf=0.2, save=True)
