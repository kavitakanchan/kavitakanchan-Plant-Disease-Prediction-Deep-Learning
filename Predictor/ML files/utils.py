import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import io

# Import our model definition
from .model import SimpleCNN

# --- 1. Load Model and Class Names ---

# Get the base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to model and class names
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
CLASSES_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load class names
with open(CLASSES_PATH) as f:
    class_names = json.load(f)
num_classes = len(class_names)

# Load the model structure
# We use map_location=device to load the model onto the correct device (CPU or GPU)
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# Set model to evaluation mode (VERY IMPORTANT)
model.eval()

# --- 2. Define Image Transforms ---
# These MUST be the same as your validation/test transforms
# From your code: IMG_SIZE = 128
IMG_SIZE = 128
prediction_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- 3. Create Prediction Function ---

def predict_image(image_bytes):
    """
    Takes image bytes, transforms it, and returns prediction.
    """
    try:
        # Open the image from bytes
        # .convert('RGB') ensures it has 3 channels
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the transformations
        # .unsqueeze(0) adds the batch dimension (1, 3, 128, 128)
        image_tensor = prediction_transform(image).unsqueeze(0).to(device)

        # Get model output
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top prediction
        # confidence, predicted_index
        confidence, preds_index = torch.max(probabilities, 1)
        
        # Get the class name
        predicted_class_name = class_names[preds_index.item()]
        
        return predicted_class_name, confidence.item()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0
