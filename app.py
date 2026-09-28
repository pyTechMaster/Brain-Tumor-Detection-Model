import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page config
st.set_page_config(
    page_title="🧠 Brain Tumor AI Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .healthy-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    .tumor-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Preprocessing function
def preprocess_image(image, target_size=128):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 255.0
    
    return np.expand_dims(image, axis=0)

# Prediction function
def predict_tumor(model, image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)
    
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]
    
    return classes[predicted_idx], confidence, prediction[0]

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor AI Detector</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an MRI scan for AI-powered analysis")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Info")
        st.info("""
        - **Architecture**: Custom CNN
        - **Input Size**: 128×128 pixels
        - **Classes**: 4 tumor types
        - **Framework**: TensorFlow
        """)
        
        st.markdown("## ⚠️ Disclaimer")
        st.warning("This tool is for educational purposes only. Always consult medical professionals.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear MRI brain scan image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="MRI Brain Scan", use_column_width=True)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            if st.button("🔍 Analyze with AI", type="primary"):
                with st.spinner("🧠 AI is analyzing..."):
                    try:
                        predicted_class, confidence, all_probs = predict_tumor(model, image)
                        
                        # Display results
                        if predicted_class == "No Tumor":
                            st.markdown(f"""
                            <div class="result-box healthy-box">
                                <h3>✅ Prediction: {predicted_class}</h3>
                                <h4>🎯 Confidence: {confidence:.1%}</h4>
                                <p>No signs of tumor detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box tumor-box">
                                <h3>⚠️ Prediction: {predicted_class}</h3>
                                <h4>🎯 Confidence: {confidence:.1%}</h4>
                                <p>Signs of {predicted_class.lower()} detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show probabilities
                        st.subheader("📊 All Probabilities")
                        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                        for i, (class_name, prob) in enumerate(zip(classes, all_probs)):
                            st.progress(float(prob), text=f"{class_name}: {prob:.1%}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()