# Train YOLO model with Roboflow annotated data set
# Use Ultralytics YOLOv11 neural network

from ultralytics import YOLO
import glob
import cv2

#Data:
data_src = "" #insert data.yaml (e.g. from roboflow annotation)

# Load a model
#model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


# Train YOLO v11 on the custom dataset
def train_yolo():
    raw_model = YOLO("yolo11n.pt")  # Load YOLOv11 pre-trained model
    model = raw_model.train(data=data_src, epochs=50, imgsz=640)
    return model


#'''
# Load the trained model
def load_model():
    return YOLO("/Users/joshkaura/runs/detect/train3/weights/last.pt")  # Update path if necessary

#single picture
def image_results(model, image="/Users/joshkaura/Desktop/OSL/AI Project/Screwdriver/jpgs/IMG_9057.jpg"):
    img = cv2.imread(image)
    results = model(image)
    for result in results:
        print(result.boxes)
    


# Perform inference on a folder of .jpg images
def images_results(model, image_folder="/Users/joshkaura/Desktop/OSL/AI Project/Screwdriver/jpgs"):
    image_paths = glob.glob(f"{image_folder}/*.jpg")

    for image_path in image_paths:
        img = cv2.imread(image_path)  # Read image
        results = model(image_path, conf=0.1)   # Get predictions

        # Draw bounding boxes on the image
        for result in results:
            for box in result.boxes:
                #print(box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                #print(x1,y1,x2,y2)
                conf = box.conf[0].item()  # Confidence score
                label = result.names[int(box.cls[0])]  # Class label
                print(box.cls)
                # Draw rectangle and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display both original and annotated image
        cv2.imshow("Original", cv2.imread(image_path))
        cv2.imshow("Annotated", img)
        cv2.waitKey(0)  # Wait for key press to move to next image

    cv2.destroyAllWindows()

#video results
def vid_results(model, vid_path = "/Users/joshkaura/Desktop/OSL/AI Project/Screwdriver/screwdriver_vid.mov"):
    """
    Runs YOLOv11 object detection on a video file and displays the results.

    """
    cap = cv2.VideoCapture(vid_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        results = model(frame, conf=0.25)  # Run detection

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
                label = f"{result.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()

def live_results(model):
    """
    Runs YOLOv11 object detection on a live webcam feed and displays results in real-time.

    """
    cap = cv2.VideoCapture(0)  # Open default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Camera issue

        results = model(frame, conf=0.2)  # Run detection

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
                label = f"{result.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Detection - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":

    #train_yolo()  # Train the model

    trained_model = load_model()  # Load the best model
    #image_results(trained_model)
    #images_results(trained_model)  # Run inference and display results
    #vid_results(trained_model)
    live_results(trained_model)

