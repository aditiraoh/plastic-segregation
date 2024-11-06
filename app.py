from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
import base64
from io import BytesIO
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the device to run the model on (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the first InceptionV3 model and its weights
model1 = models.inception_v3(pretrained=False, aux_logits=True)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 7)  # Update to 7 classes for resin codes
model1.load_state_dict(torch.load('C:\\Users\\aditi\\Desktop\\resin try\\resin try\\mp final\\inception_resin_classifier.pth', map_location=device))
model1 = model1.to(device)
model1.eval()

# Load the second InceptionV3 model and its weights
model2 = models.inception_v3(pretrained=False, aux_logits=True)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, 7)  # Update to 7 classes for resin codes
model2.load_state_dict(torch.load('C:\\Users\\aditi\\Desktop\\resin try\\resin try\\mp final\\seven_plastic_classifier.pth', map_location=device))
model2 = model2.to(device)
model2.eval()

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 expects 299x299 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict the class of an input image using both models
def predict_image(image):
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs1 = model1(image)
        outputs2 = model2(image)

        # Average the probabilities from both models
        avg_outputs = (outputs1 + outputs2) / 2
        _, preds = torch.max(avg_outputs, 1)

    return preds.item()

# Resin code labels (starting from 0)
resin_labels = {
    0: "PET",
    1: "HDPE",
    2: "PVC",
    3: "LDPE",
    4: "PP",
    5: "PS",
    6: "Other"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/pet')
def pet():
    return render_template('pet.html')

@app.route('/hdpe')
def hdpe():
    return render_template('hdpe.html')

@app.route('/pvc')
def pvc():
    return render_template('pvc.html')

@app.route('/ldpe')
def ldpe():
    return render_template('ldpe.html')

@app.route('/pp')
def pp():
    return render_template('pp.html')

@app.route('/ps')
def ps():
    return render_template('ps.html')

@app.route('/other')
def other():
    return render_template('other.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            filename = file.filename
            file_path = os.path.join('static', filename)
            
            # Ensure the static directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file.save(file_path)

            # Load image
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Convert image to grayscale
            if img.shape[2] == 4:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Initialize ORB detector
            orb = cv2.ORB_create()

            # Detect key points and descriptors
            kp, des = orb.detectAndCompute(img_gray, None)

            # Find the central region of the matches (simulate without another image for simplicity)
            if len(kp) > 0:
                pts = np.float32([k.pt for k in kp])

                # Compute the centroid of key points
                center_x, center_y = np.mean(pts, axis=0)

                # Define the size of the crop area (e.g., 200x200 pixels)
                crop_width = 200
                crop_height = 200

                # Convert padding from cm to pixels (assuming 300 DPI)
                dpi = 300
                cm_to_inch = 0.393701
                padding_cm = 2  # You can change this to 3 for 3 cm padding
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
                cropped_image_filename = f'cropped_{filename}'
                cropped_image_path = os.path.join('static', cropped_image_filename)
                cv2.imwrite(cropped_image_path, cropped_image)

                # Convert the cropped image to PIL format and classify it
                cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                class_idx = predict_image(cropped_image_pil)
                
                if class_idx >= 0 and class_idx <= 6:
                    prediction = resin_labels[class_idx]
                else:
                    prediction = "Not plastic symbol"
            else:
                prediction = "No key points found"

            logging.info(f'Prediction: {prediction}, Cropped image: {cropped_image_filename}')
            return jsonify({'prediction': prediction, 'cropped_image': cropped_image_filename})
    except Exception as e:
        logging.error(f'Error during image upload and processing: {e}')
        return jsonify({'error': 'An error occurred during processing'}), 500

@app.route('/upload_base64', methods=['POST'])
def upload_base64():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        image_data = data['image']
        image_data = image_data.split(',')[1]  # Remove the base64 header
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Save the image
        filename = 'captured_image.png'
        file_path = os.path.join('static', filename)
        image.save(file_path)

        # Convert image to grayscale
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect key points and descriptors
        kp, des = orb.detectAndCompute(img_gray, None)

        # Find the central region of the matches (simulate without another image for simplicity)
        if len(kp) > 0:
            pts = np.float32([k.pt for k in kp])

            # Compute the centroid of key points
            center_x, center_y = np.mean(pts, axis=0)

            # Define the size of the crop area (e.g., 200x200 pixels)
            crop_width = 200
            crop_height = 200

            # Convert padding from cm to pixels (assuming 300 DPI)
            dpi = 300
            cm_to_inch = 0.393701
            padding_cm = 2  # You can change this to 3 for 3 cm padding
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
            cropped_image_filename = f'cropped_{filename}'
            cropped_image_path = os.path.join('static', cropped_image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)

            # Convert the cropped image to PIL format and classify it
            cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            class_idx = predict_image(cropped_image_pil)
            
            if class_idx >= 0 and class_idx <= 6:
                prediction = resin_labels[class_idx]
            else:
                prediction = "Not plastic symbol"
        else:
            prediction = "No key points found"

        logging.info(f'Prediction: {prediction}, Cropped image: {cropped_image_filename}')
        return jsonify({'prediction': prediction, 'cropped_image': cropped_image_filename})
    except Exception as e:
        logging.error(f'Error during image upload and processing: {e}')
        return jsonify({'error': 'An error occurred during processing'}), 500

if __name__ == '__main__':
    app.run(debug=True)
