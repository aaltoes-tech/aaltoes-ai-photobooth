import streamlit as st
import os
import time
import base64
import cv2
from PIL import Image
from scripts import (
    capture_image, generate_face_mask, generate_encoding_image,
    generate_polaroid_image, upload_file, generate_qr_from_url,
    send_email, clear_images
)
from config import prompt_text, clothes_prompt, background_prompt

# Set page config with light theme
st.set_page_config(
    page_title="AI PhotoBooth", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📸",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI PhotoBooth - Created for Aaltoes"
    }
)

# Use Streamlit's built-in theming
st.markdown("""
<style>
    /* Import Geist fonts - both Sans and Mono */
    @import url('https://fonts.cdnfonts.com/css/geist-sans');
    @import url('https://fonts.cdnfonts.com/css/geist-mono');
    
    /* Apply Geist Sans to all elements */
    html, body, [class*="css"] {
        font-family: 'Geist', 'Geist Sans', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Geist', 'Geist Sans', sans-serif !important;
        font-weight: 500;
    }
    
    p, span, div, label, input, textarea, select {
        font-family: 'Geist', 'Geist Sans', sans-serif !important;
    }
    
    /* Apply Geist Mono to buttons for a technical look */
    button, .stButton button {
        font-family: 'Geist Mono', monospace !important;
        letter-spacing: 0.02em;
        font-weight: 500;
    }
    
    /* Button styling improvements */
    .stButton button {
        background-color: #0366d6;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #0353b4;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Remove fullscreen button from images */
    button[title="View fullscreen"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Set the Streamlit theme configuration
st.markdown("""
<script>
    var elements = window.parent.document.querySelectorAll('iframe');
    for (var i = 0; i < elements.length; i++) {
        doc = elements[i].contentDocument || elements[i].contentWindow.document;
        const styleElement = doc.createElement('style');
        styleElement.textContent = `
            :root {
                --background-color: #ffffff;
                --secondary-background-color: #f8f9fa;
                --primary-color: #0366d6;
                --text-color: #24292e;
                --font: 'Geist', 'Geist Sans', sans-serif;
            }
        `;
        doc.head.appendChild(styleElement);
    }
</script>
""", unsafe_allow_html=True)

# Create a config.toml file for Streamlit theming if it doesn't exist
config_dir = os.path.expanduser("~/.streamlit")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

config_path = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        f.write("""
[theme]
primaryColor = "#0366d6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#24292e"
font = "sans serif"
        """)

# Initialize session state
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "capture"  # Modes: capture, results

# Display logo centered above the title
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(
    "aaltoes_dark-3.png",
    width=200
)
st.markdown("</div>", unsafe_allow_html=True)

# App title and description (centered)
st.markdown("<h1 style='text-align: center;'>AI PhotoBooth</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center;'>
Transform your photos with AI - change backgrounds and clothing styles while preserving your face.
</p>
""", unsafe_allow_html=True)

# Sidebar with options - always visible
st.sidebar.header("Settings")

# Add model selection dropdown
ideogram_model = st.sidebar.selectbox(
    "Ideogram Model",
    [
        "ideogram-ai/ideogram-v3-turbo", 
        "ideogram-ai/ideogram-v3-balanced",
        "ideogram-ai/ideogram-v3-quality"
    ],
    index=0,
    help="Select the Ideogram model to use. Turbo is fastest, Quality is highest quality but slowest."
)

# Allow user to customize prompts
custom_clothing = st.sidebar.text_area(
    "Custom clothing style (optional)", 
    clothes_prompt,
    help="Describe the clothing style you want to apply"
)

custom_background = st.sidebar.text_area(
    "Custom background (optional)", 
    background_prompt,
    help="Describe the background setting you want"
)

# CAPTURE MODE - Show capture and transform sections
if st.session_state["view_mode"] == "capture":
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Capture Image")
        
        # Option to upload image or use webcam
        option = st.radio("Choose input method:", ["Upload image", "Use webcam"])
        
        if option == "Upload image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Create input directory if it doesn't exist
                os.makedirs("input", exist_ok=True)
                
                # Save the uploaded file
                timestamp = str(time.time())
                image_path = f"input/image_{timestamp}.jpg"
                
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.image(image_path, caption="Uploaded Image", use_column_width=True)
                st.session_state["timestamp"] = timestamp
        
        else:  # Use webcam
            # Use Streamlit's built-in camera input instead of OpenCV
            st.write("Webcam Preview:")
            
            # Use the Streamlit camera_input component for better real-time preview
            camera_image = st.camera_input("Take a photo with your webcam")
            
            if camera_image:
                # Create input directory if it doesn't exist
                os.makedirs("input", exist_ok=True)
                
                # Save the captured image
                timestamp = str(time.time())
                image_path = f"input/image_{timestamp}.jpg"
                
                with open(image_path, "wb") as f:
                    f.write(camera_image.getbuffer())
                
                # Store the timestamp in session state
                st.session_state["timestamp"] = timestamp
                
                # Display confirmation
                st.success("Photo captured successfully!")
    
    with col2:
        st.header("Transform Image")
        
        if "timestamp" in st.session_state:
            timestamp = st.session_state["timestamp"]
            
            if st.button("Transform Image"):
                with st.spinner("Processing..."):
                    try:
                        # Run the transformation pipeline
                        st.text("Generating face mask...")
                        generate_face_mask(timestamp)
                        
                        st.text("Analyzing image...")
                        encoding = generate_encoding_image(prompt_text, timestamp)
                        
                        st.text(f"Transforming image with AI using {ideogram_model} (this may take a minute)...")
                        generate_polaroid_image(
                            encoding, 
                            custom_clothing, 
                            custom_background,
                            timestamp,
                            model=ideogram_model
                        )
                        
                        st.text("Uploading to IPFS...")
                        url = upload_file(f"final/image_{timestamp}.png")
                        
                        st.text("Generating QR code...")
                        generate_qr_from_url(url, "qr_code.png")
                        
                        # Display results
                        st.success("Transformation complete!")
                        
                        # Store the result URL in session state
                        st.session_state["result_url"] = url
                        
                        # Switch to results view mode
                        st.session_state["view_mode"] = "results"
                        
                        # Force a rerun to update the UI
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        else:
            st.info("Please capture or upload an image first")

# RESULTS MODE - Show results section with the transformed image, QR code, and email option
elif st.session_state["view_mode"] == "results" and "result_url" in st.session_state and "timestamp" in st.session_state:
    timestamp = st.session_state["timestamp"]
    url = st.session_state["result_url"]
    
    # Store the final image path for later use
    final_image_path = f"final/image_{timestamp}.png"
    
    # Check if we need to clean up temp files
    if "files_cleaned" not in st.session_state:
        try:
            # Make a copy of the final image before cleaning
            os.makedirs("saved", exist_ok=True)
            saved_image_path = f"saved/image_{timestamp}.png"
            import shutil
            shutil.copy2(final_image_path, saved_image_path)
            
            # Clean the temporary files
            clear_images(timestamp)
            
            # Mark files as cleaned
            st.session_state["files_cleaned"] = True
            st.session_state["saved_image_path"] = saved_image_path
        except Exception as e:
            st.error(f"Error cleaning temporary files: {str(e)}")
    
    st.header("Your Transformed Photo")
    
    # Results container with nice layout
    results_container = st.container()
    
    with results_container:
        # Two columns layout: smaller image on left, controls on right
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the saved image with reduced width
            image_to_display = st.session_state.get("saved_image_path", final_image_path)
            st.image(
                image_to_display, 
                caption="AI-Transformed Image", 
                width=500  # Set a smaller fixed width
            )
            # Add a direct link to the image
            st.markdown(f"[View full-size image online]({url})")
            
        with col2:
            # Email section in right column
            st.subheader("Send to Email")
            user_email = st.text_input("Enter your email:")
            
            # Email send button
            if st.button("Send Email", key="send_email_button"):
                if not user_email:
                    st.warning("Please enter your email address")
                else:
                    with st.spinner("Sending email..."):
                        try:
                            send_email(user_email, st.session_state["result_url"])
                            st.success(f"Image sent to {user_email}")
                        except Exception as e:
                            st.error(f"Error sending email: {str(e)}")
            
            # Add some vertical space
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Start over button in the right column
            if st.button("Start Over", key="start_over_button", use_container_width=True):
                # Reset the session state
                st.session_state["view_mode"] = "capture"
                if "result_url" in st.session_state:
                    del st.session_state["result_url"]
                if "timestamp" in st.session_state:
                    del st.session_state["timestamp"]
                if "files_cleaned" in st.session_state:
                    del st.session_state["files_cleaned"]
                if "saved_image_path" in st.session_state:
                    del st.session_state["saved_image_path"]
                st.experimental_rerun()

# Add info about the project (only in capture mode)
if st.session_state["view_mode"] == "capture":
    st.markdown("---")
    st.markdown("""
    ### How it works
    1. Your face is detected and masked to preserve your identity
    2. AI transforms the clothing and background while keeping your face
    3. The final polaroid-style image is stored permanently on IPFS
    4. A QR code is generated for easy access to your image
    """) 