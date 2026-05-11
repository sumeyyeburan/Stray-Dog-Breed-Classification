from ultralytics import YOLO
import cv2

# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

# LOAD IMAGE
image_path = "test.jpeg"

results = model(image_path)

# READ IMAGE
image = cv2.imread(image_path)

# COCO class id for dog = 16
DOG_CLASS_ID = 16

# LOOP DETECTIONS
for result in results:

    boxes = result.boxes

    for box in boxes:

        cls = int(box.cls[0])

        # ONLY DOG
        if cls == DOG_CLASS_ID:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # CROP
            cropped_dog = image[y1:y2, x1:x2]

            # SAVE
            cv2.imwrite("cropped_dog.jpg", cropped_dog)

            print("Dog cropped and saved.")