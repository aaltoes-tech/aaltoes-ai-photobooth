import streamlit as st
import os
import time
import base64
import cv2
from PIL import Image
from scripts import (
    capture_image, generate_face_mask, generate_encoding_image,
    generate_polaroid_image, upload_file, generate_qr_from_url,
    send_email, clear_images, clean_all_images
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

# Create a config.toml file for Streamlit's default light theme
config_dir = os.path.expanduser("~/.streamlit")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

config_path = os.path.join(config_dir, "config.toml")
with open(config_path, "w") as f:
    f.write("""
[theme]
primaryColor = "#0366d6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f5"
textColor = "#262730"
font = "sans serif"
    """)

# Remove all custom CSS styling - use Streamlit defaults

# Clean up any leftover images from previous sessions
clean_all_images()

# Also edit the scripts.py file to fix the MTCNN error with MPS
def fix_scripts_py():
    try:
        scripts_file = "scripts.py"
        with open(scripts_file, "r") as f:
            content = f.read()
            
        # Fix the MPS error in the MTCNN initialization
        if "device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'" in content:
            content = content.replace(
                "device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'",
                "device='cuda' if torch.cuda.is_available() else 'cpu'"  # Always use CPU as fallback, not MPS
            )
            
            with open(scripts_file, "w") as f:
                f.write(content)
            print("Fixed MTCNN device in scripts.py")
    except Exception as e:
        print(f"Couldn't fix scripts.py: {e}")

# Try to fix the scripts.py file
fix_scripts_py()

# Initialize session state
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "capture"  # Modes: capture, results

# Add a state to track if transformation is in progress
if "transforming" not in st.session_state:
    st.session_state["transforming"] = False

# Function to start transformation
def start_transformation():
    st.session_state["transforming"] = True

# Function to process webcam image
def process_webcam_image(webcam_image):
    # Create input directory if it doesn't exist
    os.makedirs("input", exist_ok=True)
    
    # Save the captured image
    timestamp = str(time.time())
    image_path = f"input/image_{timestamp}.jpg"
    
    with open(image_path, "wb") as f:
        f.write(webcam_image.getbuffer())
    
    # Store the timestamp in session state
    st.session_state["timestamp"] = timestamp
    
    # Display confirmation
    st.success("Photo captured successfully!")

# Header section with logo and title - only show in capture mode
# Sidebar with options - always visible
with st.sidebar:
    st.image(
        "aaltoes_dark-3.png",
        width=180
    )
    st.header("Settings")
    
    # Add model selection dropdown with a cleaner look
    st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    ideogram_model = st.selectbox(
        "Ideogram Model",
        [
            "ideogram-ai/ideogram-v3-turbo", 
            "ideogram-ai/ideogram-v3-balanced",
            "ideogram-ai/ideogram-v3-quality"
        ],
        index=0,
        help="Select the Ideogram model to use. Turbo is fastest, Quality is highest quality but slowest."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Allow user to customize prompts
    st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.subheader("Customization")
    custom_clothing = st.text_area(
        "Clothing style", 
        clothes_prompt,
        help="Describe the clothing style you want to apply"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    custom_background = st.text_area(
        "Background setting", 
        background_prompt,
        help="Describe the background setting you want"
    )
    st.markdown('</div>', unsafe_allow_html=True)

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
            
            # Simple webcam capture without timer functionality
            camera_image = st.camera_input("Take a photo with your webcam")
            
            # Process captured image
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
            
            # Show the captured image thumbnail if available
            image_path = f"input/image_{timestamp}.jpg"
            if os.path.exists(image_path):
                st.write("Ready to transform this image:")
                st.image(image_path, width=300)
            
            # Only show the transform button if not currently transforming
            if not st.session_state["transforming"]:
                if st.button("Transform Image", key="transform_button", on_click=start_transformation, use_container_width=True):
                    pass  # The on_click function will set the transforming state to True
            
            # Show processing if transformation is in progress
            if st.session_state["transforming"]:
                with st.spinner("Processing..."):
                    try:
                        # Create a progress indicator
                        progress_placeholder = st.empty()
                        progress_placeholder.info("Starting image transformation process...")
                        
                        # Run the transformation pipeline
                        progress_placeholder.info("Detecting faces and generating face mask (this may take 10-15 seconds)...")
                        try:
                            generate_face_mask(timestamp)
                            progress_placeholder.success("Face mask created successfully")
                        except Exception as e:
                            # Show warning but continue processing
                            st.warning(f"Face detection had issues: {str(e)}")
                            progress_placeholder.warning("Simplified face mask will be used")
                            # Don't raise, just continue
                        
                        progress_placeholder.info("Analyzing image with AI...")
                        encoding = generate_encoding_image(prompt_text, timestamp)
                        progress_placeholder.success("Image analysis complete")
                        
                        progress_placeholder.info(f"Transforming image with AI using {ideogram_model} (this may take a minute)...")
                        generate_polaroid_image(
                            encoding, 
                            custom_clothing, 
                            custom_background,
                            timestamp,
                            model=ideogram_model
                        )
                        progress_placeholder.success("Image transformation complete")
                        
                        progress_placeholder.info("Uploading to IPFS...")
                        url = upload_file(f"final/image_{timestamp}.png")
                        progress_placeholder.success("Image uploaded successfully")
                        
                        progress_placeholder.info("Generating QR code...")
                        generate_qr_from_url(url, "qr_code.png")
                        progress_placeholder.success("QR code generated")
                        
                        # Display results
                        progress_placeholder.success("Transformation complete!")
                        
                        # Store the result URL in session state
                        st.session_state["result_url"] = url
                        
                        # Reset transforming state before switching views
                        st.session_state["transforming"] = False
                        
                        # Switch to results view mode
                        st.session_state["view_mode"] = "results"
                        
                        # Force a rerun to update the UI
                        st.rerun()
                        
                    except Exception as e:
                        # Reset transforming state on error
                        st.session_state["transforming"] = False
                        st.error(f"Error during processing: {str(e)}")
                        # Force rerun to show the button again
                        st.rerun()
        else:
            st.info("Please capture or upload an image first")

# RESULTS MODE - Show results section with the transformed image, QR code, and email option
elif st.session_state["view_mode"] == "results" and "result_url" in st.session_state and "timestamp" in st.session_state:
    timestamp = st.session_state["timestamp"]
    url = st.session_state["result_url"]
    
    # Clean up temp files immediately
    if "files_cleaned" not in st.session_state:
        try:
            # Clean all temporary files - we'll display from URL
            clear_images(timestamp, keep_final=False)
            st.session_state["files_cleaned"] = True
        except Exception as e:
            # Just log the error but don't show to user - non-critical
            print(f"Error cleaning temporary files: {e}")
            # Mark as cleaned anyway to avoid repeated attempts
            st.session_state["files_cleaned"] = True
    
    # Container for transformed image
    st.header("Your Transformed Photo")
    
    # Center the image with Streamlit columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Display image directly from URL instead of local file
        st.image(
            url, 
            caption="", 
            use_column_width=True
        )
        
        # Start over button right under the image
        if st.button("Start Over", key="start_over_button", use_container_width=True):
            # Reset the session state
            st.session_state["view_mode"] = "capture"
            st.session_state["transforming"] = False  # Reset the transforming state
            if "result_url" in st.session_state:
                del st.session_state["result_url"]
            if "timestamp" in st.session_state:
                del st.session_state["timestamp"]
            if "files_cleaned" in st.session_state:
                del st.session_state["files_cleaned"]
            
            # Clean all images when starting over
            clean_all_images()
            
            st.rerun()
    
    # Email section
    st.subheader("Send to Email")
    
    # Center the email input and buttons
    email_col1, email_col2, email_col3 = st.columns([1, 2, 1])
    
    with email_col2:
        # Remove placeholder text so "press enter to apply" doesn't show
        user_email = st.text_input("", value="", placeholder="Your email address", key="email_input")
        
        # Email send button
        if st.button("Send Email", key="send_email_button", use_container_width=True):
            if not user_email:
                st.warning("Please enter your email address")
            else:
                with st.spinner("Sending email..."):
                    try:
                        send_email(user_email, st.session_state["result_url"])
                        st.success(f"Image sent to {user_email}")
                    except Exception as e:
                        st.error(f"Error sending email: {str(e)}")

# Add info about the project (only in capture mode)
if st.session_state["view_mode"] == "capture":
    st.subheader("How it works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 1. Face Detection
        Your face is detected and masked to preserve your identity
        """)
    
    with col2:
        st.markdown("""
        #### 2. AI Transformation
        AI transforms the clothing and background while keeping your face
        """)
    
    with col3:
        st.markdown("""
        #### 3. Image Storage
        The final polaroid-style image is stored permanently on IPFS
        """)
    
    with col4:
        st.markdown("""
        #### 4. QR Code Generation
        A QR code is generated for easy access to your image
        """) 