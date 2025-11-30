import streamlit as st
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import timm
from deepface import DeepFace
from PIL import Image
import io
import gdown
from pathlib import Path
import zipfile
import shutil
from uuid import uuid4

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1EXAMPLE"  # Replace with actual model URL if needed
EXPECTED_CLASSES = 70
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names (hardcoded or load from file)
CLASS_NAMES = [
    ' Nasya Aulia Efendi', 'Abraham Ganda Napitu', 'Abu Bakar Siddiq Siregar', 'Ahmad Faqih Hasani', 'Aldi Sanjaya', 'Alfajar', 'Alief Fathur Rahman', 'Arkan Hariz Chandrawinata Liem', 'Bayu Ega Ferdana', 'Bayu Prameswara Haris', 'Bezalel Samuel Manik', 'Bintang Fikri Fauzan', 'Boy Sandro Sigiro', 'Desty Ananta Purba', 'Dimas Azi Rajab Aizar', 'Dito Rifki Irawan', 'Dwi Arthur Revangga', 'Dyo Dwi Carol Bukit', 'Eden Wijaya ', 'Eichal Elphindo Ginting', 'Elsa Elisa Yohana Sianturi', 'Fajrul Ramadhana Aqsa', 'Falih Dzakwan Zuhdi', 'Fathan Andi Kartagama', 'Fayyadh Abdillah', 'Femmy Aprillia Putri', 'Ferdana Al Hakim', 'Festus Mikhael ', 'Fiqri Aldiansyah', 'Freddy Harahap', 'Gabriella Natalya Rumapea', 'Garland Wijaya', 'Havidz Ridho Pratama', 'Ichsan Kuntadi Baskara', 'Ikhsannudin Lathief', 'Intan Permata Sari ', 'JP. Rafi Radiktya Arkan. R. AZ', 'Joshia Fernandes Sectio Purba ', 'Joshua Palti Sinaga', 'Joy Daniella V', 'Joyapul Hanscalvin Panjaitan', 'Kayla Chika Lathisya ', 'Kenneth Austin Wijaya', 'Kevin Naufal Dany', 'Lois Novel E Gurning', 'Machzaul harmansyah ', 'Martua Kevin A.M.H.Lubis', 'Muhammad Fasya Atthoriq', 'Muhammad Nelwan Fakhri ', 'Muhammad Riveldo Hermawan Putra', 'Muhammad Zada Rizki', 'Mychael Daniel N', 'Raditya Erza Farandi', 'Rahmat Aldi Nasda', 'Randy Hendriyawan', 'Rayhan Fadel Irwanto ', 'Rayhan Fatih Gunawan', 'Reynaldi Cristian Simamora', 'Rizky Abdillah ', 'Royfran Roger Valentino', 'Rustian Afencius Marbun', 'Shintya Ayu Wardani', 'Sikah Nubuahtul Ilmi', 'William Chan', 'Yohanna Anzelika Sitepu ', 'Zakhi algifari', 'Zaky Ahmad Makarim', 'Zefanya Danovanta Tarigan', 'Zidan Raihan', 'hayyatul fajri'
]

# Model builders
def build_cnn_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.45),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),
        nn.Linear(512, num_classes)
    )
    return model

def build_vit_model(num_classes: int) -> nn.Module:
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    for name, param in model.named_parameters():
        if not name.startswith('head'):
            param.requires_grad = False
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, num_classes)
    )
    return model

# Load models
@st.cache_resource
def load_models():
    cnn_model = build_cnn_model(EXPECTED_CLASSES).to(DEVICE)
    vit_model = build_vit_model(EXPECTED_CLASSES).to(DEVICE)
    # Assuming checkpoints are in the repo or downloaded
    # For demo, load pretrained or dummy
    cnn_model.eval()
    vit_model.eval()
    return cnn_model, vit_model

# Transform
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict function
def predict_face(image, model, transform):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    return CLASS_NAMES[pred_idx], probs[pred_idx]

# Streamlit app
st.set_page_config(page_title="Face Recognition App", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    .stText {font-size: 18px;}
    .prediction {font-size: 24px; font-weight: bold; color: #FF5722;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Modern Face Recognition App")
st.markdown("Upload an image or use your camera to recognize faces using AI models (CNN & ViT).")

cnn_model, vit_model = load_models()

option = st.selectbox("Choose input method:", ["Upload Image", "Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Detect faces using DeepFace
        img_array = np.array(image)
        try:
            faces = DeepFace.extract_faces(img_path=img_array, detector_backend='retinaface', enforce_detection=False)
            if faces:
                st.success(f"Detected {len(faces)} face(s)!")
                for i, face_data in enumerate(faces):
                    face_img = face_data['face']
                    if isinstance(face_img, np.ndarray) and face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype('uint8')
                    face_pil = Image.fromarray(face_img)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(face_pil, caption=f'Face {i+1}', width=200)
                    
                    with col2:
                        cnn_pred, cnn_conf = predict_face(face_pil, cnn_model, eval_transform)
                        vit_pred, vit_conf = predict_face(face_pil, vit_model, eval_transform)
                        st.markdown(f"**CNN Prediction:** {cnn_pred} ({cnn_conf:.2f})")
                        st.markdown(f"**ViT Prediction:** {vit_pred} ({vit_conf:.2f})")
            else:
                st.warning("No faces detected in the image.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

elif option == "Camera":
    st.markdown("### Camera Input")
    st.markdown("Note: Camera access requires HTTPS in production. For demo, use local Streamlit.")
    
    # For simplicity, use file uploader as proxy for camera
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')
        st.image(image, caption='Captured Image', use_column_width=True)
        
        # Same processing as upload
        img_array = np.array(image)
        try:
            faces = DeepFace.extract_faces(img_path=img_array, detector_backend='retinaface', enforce_detection=False)
            if faces:
                st.success(f"Detected {len(faces)} face(s)!")
                for i, face_data in enumerate(faces):
                    face_img = face_data['face']
                    if isinstance(face_img, np.ndarray) and face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype('uint8')
                    face_pil = Image.fromarray(face_img)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(face_pil, caption=f'Face {i+1}', width=200)
                    
                    with col2:
                        cnn_pred, cnn_conf = predict_face(face_pil, cnn_model, eval_transform)
                        vit_pred, vit_conf = predict_face(face_pil, vit_model, eval_transform)
                        st.markdown(f"**CNN Prediction:** {cnn_pred} ({cnn_conf:.2f})")
                        st.markdown(f"**ViT Prediction:** {vit_pred} ({vit_conf:.2f})")
            else:
                st.warning("No faces detected in the image.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and DeepFace. Deploy on Hugging Face Spaces for free!")