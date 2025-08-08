import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

# -----------------------
# Define class names here
# -----------------------
class_names = ['airplane', 'cat', 'bird', 'deer', 'automobile']  # Change based on dataset

# -----------------------
# Define Custom CNN class
# -----------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Automatically calculate the in_features for the first Linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -----------------------
# Model loader function
# -----------------------
def load_model(model_name, num_classes):
    if model_name == "Custom_cnn":
        model = CustomCNN(num_classes)
        try:
            model.load_state_dict(torch.load("custom_cnn.pth", map_location=torch.device('cpu')))
        except RuntimeError as e:
            print(f"[WARNING] State dict mismatch detected: {e}")
            print("[INFO] Loading with strict=False (some layers will be randomly initialized).")
            state_dict = torch.load("custom_cnn.pth", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)

    elif model_name == "VGG16_custom":
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, num_classes)
        try:
            model.load_state_dict(torch.load("vgg16_custom.pth", map_location=torch.device('cpu')))
        except RuntimeError as e:
            print(f"[WARNING] {e}")
            state_dict = torch.load("vgg16_custom.pth", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)

    elif model_name == "resNet50_custom":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        try:
            model.load_state_dict(torch.load("resnet50_custom.pth", map_location=torch.device('cpu')))
        except RuntimeError as e:
            print(f"[WARNING] {e}")
            state_dict = torch.load("resnet50_custom.pth", map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

# -----------------------
# Streamlit App UI
# -----------------------
st.title("🖼️ Image Classification App")
model_name = st.selectbox("Choose model:", ["Custom_cnn", "VGG16_custom", "resNet50_custom"])

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_name:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(image).unsqueeze(0)
    model = load_model(model_name, num_classes=5)  # Update num_classes based on dataset
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

    st.markdown("### 🔍 Prediction")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Predicted Class:** {class_names[pred.item()]}")
    st.write(f"**Confidence:** {conf.item() * 100:.2f}%")
