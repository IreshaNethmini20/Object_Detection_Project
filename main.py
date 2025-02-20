import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Getting Output Layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load and preprocess the image
image = cv2.imread("data/images/bus&people.jpg")   
if image is None:
    raise FileNotFoundError("Image file not found. Check the path.")

# Resize image to fit screen size
screen_width, screen_height = 800, 600  # Adjust this based on your screen resolution
image = cv2.resize(image, (screen_width, screen_height))

height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# Process detection results
boxes, confidences, class_ids = [], [], []
object_counts = {}  # Dictionary to store object categories

# Apply Non-Maximum Suppression (NMS)   
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5: #Ignore Low-Confidence Detections
            center_x, center_y, w, h = ( #Convert Box Position to Pixels
                int(detection[0] * width),
                int(detection[1] * height),
                int(detection[2] * width),
                int(detection[3] * height),
            )
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            boxes.append([x, y, w, h]) # Store the Object Details
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) (Remove Duplicate Boxes )
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw Boxes and Labels
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Count Objects
        object_name = classes[class_ids[i]]
        object_counts[object_name] = object_counts.get(object_name, 0) + 1

# Print categorized objects and their counts
print("Detected Object Categories:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")

# Show the image with detections
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
