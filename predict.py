# predict.py
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Load models
def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained=False, aux_logits=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model1 = load_model('C:\\Users\\anura\\Desktop\\resin try\\mp final\\inception_resin_classifier.pth')
model2 = load_model('C:\\Users\\anura\\Desktop\\resin try\\mp final\\seven_plastic_classifier.pth')

data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = data_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs1 = model1(image)
        outputs2 = model2(image)
        avg_outputs = (outputs1 + outputs2) / 2
        _, preds = torch.max(avg_outputs, 1)

    return preds.item()

resin_labels = {
    0: "PET",
    1: "HDPE",
    2: "PVC",
    3: "LDPE",
    4: "PP",
    5: "PS",
    6: "Other"
}

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img_gray, None)

    reference_img_path = 'C:\\Users\\anura\\Desktop\\resin try\\mp final\\temppp.jpg'
    ref_img = cv2.imread(reference_img_path, cv2.IMREAD_UNCHANGED)
    if ref_img.shape[2] == 4:
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2GRAY)
    else:
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(ref_img_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 0:
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute the centroid of matched points
        center_x, center_y = np.mean(pts2, axis=0)

    # Define the size of the crop area (e.g., 200x200 pixels)
        crop_width = 200
        crop_height = 200

    # Convert padding from cm to pixels (assuming 300 DPI)
        dpi = 300
        cm_to_inch = 0.393701
        padding_cm = 1  # You can change this to 3 for 3 cm padding
        padding_pixels = int(padding_cm * cm_to_inch * dpi)

    # Calculate the bounding box with padding
        x_min = int(center_x - crop_width // 2 - padding_pixels)
        y_min = int(center_y - crop_height // 2 - padding_pixels)
        x_max = int(center_x + crop_width // 2 + padding_pixels)
        y_max = int(center_y + crop_height // 2 + padding_pixels)

    # Ensure the coordinates are within the image boundaries
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, img.shape[1])
        y_max = min(y_max, img.shape[0])

    # Crop the image
        cropped_image = img[y_min:y_max, x_min:x_max]

        # Save the cropped image
        cropped_image_path = 'cropped_image.jpg'
        cv2.imwrite(cropped_image_path, cropped_image)

        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        class_idx = predict_image(cropped_image_pil)

        if class_idx >= 0 and class_idx <= 6:
            return resin_labels[class_idx], cropped_image_path
        else:
            return "Not plastic symbol", cropped_image_path
    else:
        return "No matches found", None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prediction, cropped_image_path = process_image(image_path)
    print(prediction)
    if cropped_image_path:
        print(f"Cropped image saved at: {cropped_image_path}")
