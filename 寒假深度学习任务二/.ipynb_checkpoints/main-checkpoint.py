from ultralytics import YOLO
from ultralytics import settings

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# print settings
print(settings)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='dataset.yaml', epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')