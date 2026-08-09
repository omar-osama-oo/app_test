import streamlit as st
from transformers import pipeline
from PIL import Image

# Page setup
st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("🖼️ Image Classification App")
st.write("Upload an image to classify its content using Vision Transformer (ViT).")

# Cache model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)    
    st.write("Classifying...")
    
    # Run prediction
    predictions = classifier(image)
    
    # Display results
    st.subheader("Top Predictions:")
    for item in predictions:
        label = item['label']
        score = round(item['score'] * 100, 2)
        st.write(f"**{label}**: {score}%")
        st.progress(item['score'])
