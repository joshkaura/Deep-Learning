import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np

# Pre-Trained with COCO dataset -----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pre-trained model
model = torch.hub.load('pytorch/vision', 'ssdlite320_mobilenet_v3_large', pretrained=True)
model.eval()  # Set to evaluation mode

# Load labels (COCO classes)
coco_labels = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
               "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
               "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
               "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
               "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
               "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush"]

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize to SSD input size
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Object Detection Function
def detect_objects(image_path):
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        predictions = model(image)
    
    # Extract boxes and labels
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    # Load original image for display
    img = cv2.imread(image_path)
    
    # Draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i]
            label = coco_labels[labels[i]]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_objects("test.jpg")


# Model Training for custom dataset -----------------------

# Dataset Loader - Custom Dataset
class WeldingGapDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        ann_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.txt'))

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load bounding box annotations
        boxes = []
        labels = []
        with open(ann_path, "r") as f:
            for line in f.readlines():
                x_min, y_min, x_max, y_max, label = map(float, line.strip().split())
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(label))

        # Convert to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# Define Model
model = ssdlite320_mobilenet_v3_large(pretrained=True)

# Modify the classification head for welding gap detection (1 class + background)
num_classes = 2  # Welding Gap + Background
model.head.classification_head = SSDClassificationHead(in_channels=288, num_anchors=6, num_classes=num_classes)

# Define Training Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Load Custom Dataset
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

dataset = WeldingGapDataset(image_dir="data/images", annotation_dir="data/labels", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train Model
for epoch in range(50):  # Adjust number of epochs
    model.train()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "mobilenet_custom.pth")

# Load trained model
model.load_state_dict(torch.load("mobilenet_custom.pth"))
model.eval()

# Inference
def detect_welding_gap(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract boxes and labels
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Load original image
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, "Welding Gap", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Welding Gap Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example
detect_welding_gap("test.jpg")