import streamlit as st
from PIL import Image
import os
import json
import shutil
from predict import main as predict_main  # Import the main function from predict.py
import base64
import gdown

# Ensure temp and results directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Streamlit UI Configuration
st.set_page_config(
    page_title="ENTS - Weld Vision",
    page_icon="⚙️",
    layout="wide",
)

# Function to download model from Google Drive
def download_model_from_gdrive(model_path, gdrive_url):
    if not os.path.exists(model_path):
        st.info("Downloading model file...")
        try:
            gdown.download(gdrive_url, model_path, quiet=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")

# Google Drive URL for the model file
gdrive_url = "https://drive.google.com/uc?id=1JZOxFww6rKGmeanYdxKQFeCb2yfpqFii"  # Replace FILE_ID with the actual file ID from the shareable link
model_path = "model_pt.pt"

# Download the model if not already present
download_model_from_gdrive(model_path, gdrive_url)

# Convert the Tenorite font to Base64
def load_font_base64(font_path):
    with open(font_path, "rb") as f:
        encoded_font = base64.b64encode(f.read()).decode("utf-8")
    return f"data:font/ttf;base64,{encoded_font}"

# Load the Tenorite font
tenorite_font_base64 = load_font_base64("Tenorite.ttf")

# Custom CSS to Apply Global Styling and Font
st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'Tenorite';
        src: url("{tenorite_font_base64}") format('truetype');
    }}

    /* Apply Tenorite Font Globally */
    html, body, [class*="css"] {{
        font-family: 'Tenorite', sans-serif !important;
        background-color: #121212;
        color: #E0E0E0;
    }}

    /* Also ensure Streamlit default components use Tenorite */
    .stButton button, .stFileUploader, h1, h2, h3, h4, h5, h6, p, div {{
        font-family: 'Tenorite', sans-serif !important;
    }}

    /* Hero Section */
    .hero {{
        text-align: center;
        padding: 150px 20px;
        color: white;
    }}
    .hero h1 {{
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .hero h2 {{
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 20px;
    }}
    .hero .cta {{
        background-color: #5A9FFF;
        color: white;
        padding: 15px 30px;
        font-size: 1rem;
        border: none;
        border-radius: 25px;
        text-decoration: none;
    }}
    .hero .cta:hover {{
        background-color: #3B8EDC;
    }}

    /* File Uploader */
    .stFileUploader {{
        border: 2px dashed #5A9FFF;
        border-radius: 12px;
        padding: 20px;
        background-color: #1E1E26;
        color: #E0E0E0;
    }}

    /* Buttons */
    .stButton button {{
        background: #5A9FFF;
        color: white;
        padding: 10px 30px;
        border-radius: 20px;
        font-size: 1.1rem;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .stButton button:hover {{
        background: #3B8EDC;
    }}

    /* Results Card */
    .results-card {{
        background-color: #1E1E26;
        border: 1px solid #44475A;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.4);
    }}
    .results-card h3 {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #5A9FFF;
        margin-bottom: 10px;
    }}
    .results-card p {{
        font-size: 1rem;
        color: #E0E0E0;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        margin-top: 50px;
        font-size: 0.9rem;
        color: #B0B0B0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div class="hero">
        <h1>⚙️ Weld Vision</h1>
        <h2>AI-Powered Defect Detection</h2>
        <a href="#file-uploader" class="cta">Upload Your Image</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# File Upload Section
st.markdown('<div id="file-uploader"></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a Weld Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_path = os.path.join("temp", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    try:
        predictions = predict_main(image_path, threshold=0.5, verbose=False)

        result_folder = "output_result"
        base_name = os.path.splitext(uploaded_file.name)[0]
        result_image_path = os.path.join(result_folder, f"{base_name}_bbox.jpg")
        result_json_path = os.path.join(result_folder, f"r_{base_name}.json")

        if os.path.exists(result_image_path):
            # Centered heading for detection results
            st.markdown(
                "<div style='text-align:center; margin-top:80px;'><h2 style='font-size:2.5rem; font-weight:bold; margin-bottom:50px; color:#E0E0E0;'>Detection Result</h2></div>",
                unsafe_allow_html=True,
            )

            # Display images side-by-side
            col1, col2 = st.columns(2, gap="large")
            with col1:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_container_width=True)

            with col2:
                result_image = Image.open(result_image_path)
                st.image(result_image, caption="Processed Image with Bounding Boxes", use_container_width=True)

            # Display Detection Summary
            if os.path.exists(result_json_path):
                with open(result_json_path, "r") as f:
                    prediction_data = json.load(f)

                st.markdown("<h3>Detection Summary</h3>", unsafe_allow_html=True)
                for item in prediction_data:
                    class_name = item.get("class", "N/A")
                    confidence = item.get("score", 0)

                    confidence_color = (
                        "#03DAC6" if confidence >= 0.70
                        else "#FBC02D" if confidence >= 0.5
                        else "#CF6679"
                    )

                    st.markdown(
                        f"""
                        <div class='results-card'>
                            <h3>Prediction:</h3>
                            <p><strong>Class:</strong> {class_name}</p>
                            <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence * 100:.2f}%</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Prediction details not found.")
        else:
            st.error("Processed image not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer Section
st.markdown(
    """
    <div class="footer">
        Weld Vision | Powered by AI - ENTS
    </div>
    """,
    unsafe_allow_html=True,
)
