from ultralytics import YOLO

# MODEL
model = YOLO("yolov8n.pt")

# IMAGE
results = model("test.jpeg")

# SHOW RESULT
results[0].show()

# SAVE RESULT
results[0].save(filename="result.jpg")

print("Detection completed.")