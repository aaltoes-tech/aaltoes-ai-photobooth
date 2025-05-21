import streamlit as st
import cv2
from datetime import datetime
import os
from scripts import *
from config import *
import time
import base64
from PIL import Image, ImageEnhance
import numpy as np
import re

st.set_page_config(
    page_title="Aaltoes AI Photobooth",
    page_icon="📸",
    layout="centered"
)

def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 'start'
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = None
    if 'image_url' not in st.session_state:
        st.session_state.image_url = None
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'frame' not in st.session_state:
        st.session_state.frame = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "ideogram-ai/ideogram-v3-turbo"
    if 'photo_source' not in st.session_state:
        st.session_state.photo_source = None  # 'webcam' or 'upload'
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = 0  # Default to first style


def get_camera():
    if st.session_state.camera is None:
        try:
            # Use AVFoundation backend with specific settings
            camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not camera.isOpened():
                st.error("Could not open webcam. Please make sure your webcam is connected and not in use by another application.")
                return None
            
            
            st.session_state.camera = camera
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
            return None
    return st.session_state.camera

def release_camera():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

def capture_image():
    camera = get_camera()
    if camera is None:
        return None
    
    # Create a placeholder for the webcam feed
    webcam_placeholder = st.empty()
    capture_button = st.button("Capture Photo")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from webcam")
                release_camera()
                return None
            
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert frame to RGB for Streamlit and store in session state
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.frame = frame_rgb
            
            # Display the frame using session state
            webcam_placeholder.image(st.session_state.frame, channels="RGB", use_column_width=True)
            
            # Trigger autofocus periodically
            camera.set(cv2.CAP_PROP_FOCUS, 0)  # Reset focus to trigger autofocus
            
            if capture_button:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.timestamp = timestamp
                
                # Ensure input directory exists
                os.makedirs('input', exist_ok=True)
                
                # Save the raw image without any effects
                image_path = f'input/image_{timestamp}.jpg'
                success = cv2.imwrite(image_path, frame)
                
                if success and os.path.exists(image_path):
                    release_camera()
                    return timestamp
                else:
                    st.error("Failed to save the captured image")
                    return None
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
            
    except Exception as e:
        st.error(f"Error during webcam capture: {str(e)}")
        release_camera()
        return None

def main():
    initialize_session_state()
  
    st.title("Aaltoes AI Photobooth 📸")
    
    if st.session_state.step == 'start':
        st.write("Welcome to the AI Photobooth! Let's create your unique photo.")
        
        # Style selection
        style_names = [
            "Bioluminescent Explorer",
            "Workshop Inventor",
            "Cosmic Scholar",
            "Nocturnal Scientist"
        ]
        st.session_state.selected_style = st.radio(
            "Choose your style:",
            range(len(style_names)),
            format_func=lambda x: style_names[x],
            horizontal=True
        )
        
        # Model selection
        st.write("Choose an AI model for transformation:")
        models = {
            "ideogram-ai/ideogram-v3-turbo": "Fastest processing with good quality (Recommended for quick results)",
            "ideogram-ai/ideogram-v3-balanced": "Balanced speed and quality (Good for most use cases)",
            "ideogram-ai/ideogram-v3-quality": "Highest quality but slower processing (Best for final results)"
        }
        
        selected_model = st.radio(
            "Select model:",
            options=list(models.keys()),
            format_func=lambda x: f"{x.split('/')[-1].replace('ideogram-v3-', '').title()} - {models[x]}",
            index=0
        )
        st.session_state.selected_model = selected_model
        
        # Add option to choose between webcam and upload
        capture_method = st.radio(
            "Choose how to take your photo:",
            ["Use Webcam", "Upload Photo"],
            help="You can either take a photo using the webcam or upload an existing photo."
        )
        
        if capture_method == "Upload Photo":
            uploaded_file = st.file_uploader("Upload your photo", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                # Create timestamp for the uploaded file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.timestamp = timestamp
                st.session_state.photo_source = 'upload'  # Set photo source
                
                # Ensure input directory exists
                os.makedirs('input', exist_ok=True)
                
                # Save and resize the uploaded file
                image_path = f'input/image_{timestamp}.jpg'
                
                # Read image using PIL and preserve original orientation
                image = Image.open(uploaded_file)
                # Remove EXIF rotation
                if hasattr(image, '_getexif') and image._getexif() is not None:
                    exif = image._getexif()
                    if exif is not None:
                        for orientation in [274, 274]:  # EXIF orientation tag
                            if orientation in exif:
                                if exif[orientation] == 3:
                                    image = image.rotate(180, expand=True)
                                elif exif[orientation] == 6:
                                    image = image.rotate(270, expand=True)
                                elif exif[orientation] == 8:
                                    image = image.rotate(90, expand=True)
                                break
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Calculate dimensions for centered crop
                target_width, target_height = 1080, 720
                width, height = image.size
                
                # Calculate the crop dimensions
                crop_width = min(width, int(height * (target_width / target_height)))
                crop_height = min(height, int(width * (target_height / target_width)))
                
                # Calculate the crop box (centered)
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                
                # Crop and resize to target size
                image = image.crop((left, top, right, bottom))
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Save the resized image without EXIF data
                image.save(image_path, 'JPEG', quality=95, exif=b'')
                
                st.session_state.step = 'preview'
                st.rerun()
        else:
            if st.button("Start Photo Session"):
                # Test webcam access before proceeding
                test_camera = cv2.VideoCapture(0)
                if not test_camera.isOpened():
                    st.error("Could not access webcam. Please make sure your webcam is connected and not in use by another application.")
                else:
                    test_camera.release()
                    st.session_state.photo_source = 'webcam'  # Set photo source
                    st.session_state.step = 'capture'
                    st.rerun()
            
    elif st.session_state.step == 'capture':
        st.write("Position yourself in front of the camera and click 'Capture Photo' when ready!")
        st.write("If you don't see the webcam feed, please check that your webcam is connected and not in use by another application.")
        
        timestamp = capture_image()
        
        if timestamp:
            st.session_state.step = 'preview'
            st.rerun()
            
    elif st.session_state.step == 'preview':
        st.write("Here's your photo! How does it look?")
        image_path = f'input/image_{st.session_state.timestamp}.jpg'
        st.image(image_path, use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Show different button text based on photo source
            if st.session_state.photo_source == 'upload':
                if st.button("Upload Different Photo"):
                    st.session_state.step = 'start'
                    st.rerun()
            else:
                if st.button("Retake Photo"):
                    st.session_state.step = 'capture'
                    st.rerun()
        with col2:
            if st.button("Continue"):
                st.session_state.step = 'processing'
                st.rerun()
                
    elif st.session_state.step == 'processing':
        st.write("Processing your photo... This may take a few moments.")
        
        # Create a container for progress updates
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            timer_text = st.empty()  # Add container for timer
        
        # Create containers for intermediate results
        results_container = st.container()
        
        # Start the timer
        start_time = time.time()
        
        # Generate face mask
        status_text.text("Step 1/4: Generating face mask...")
        progress_bar.progress(25)
        try:
            generate_face_mask(st.session_state.timestamp)
            elapsed_time = time.time() - start_time
            timer_text.text(f"Time elapsed: {elapsed_time:.1f} seconds")
            with results_container:
                st.write("Face mask generated:")
                st.image(f'masks/image_{st.session_state.timestamp}.png', use_column_width=True)
        except ValueError as e:
            st.error(str(e))
            if st.button("Try Again with a Different Photo"):
                st.session_state.step = 'start'
                st.rerun()
            return
        
        # Generate encoding and transform image
        status_text.text("Step 2/4: Analyzing image and generating description...")
        progress_bar.progress(50)
        encoding_image = generate_encoding_image(prompt_text + hair_prompt[st.session_state.selected_style], st.session_state.timestamp)
        elapsed_time = time.time() - start_time
        timer_text.text(f"Time elapsed: {elapsed_time:.1f} seconds")
        with results_container:
            st.write("Image analysis complete. Description:")
            st.info(encoding_image)
        
        status_text.text("Step 3/4: Transforming image...")
        progress_bar.progress(75)
        generate_polaroid_image(
            encoding_image, 
            clothes_prompt[st.session_state.selected_style], 
            background_prompt[st.session_state.selected_style], 
            f"references/reference_{st.session_state.selected_style}.png",
            st.session_state.timestamp,
            model=st.session_state.selected_model,  # Use the selected model
        )
        elapsed_time = time.time() - start_time
        timer_text.text(f"Time elapsed: {elapsed_time:.1f} seconds")
        with results_container:
            st.write("Transformed image:")
            st.image(f'stacked/image_{st.session_state.timestamp}.jpg', use_column_width=True)
        
        # Upload and generate QR code
        status_text.text("Step 4/4: Finalizing and uploading...")
        st.session_state.image_url = upload_file(f"final/image_{st.session_state.timestamp}.png")
        clear_images(st.session_state.timestamp)
        
        # Generate QR code with timestamp
        qr_path = f'qr/qr_code_{st.session_state.timestamp}.png'
        generate_qr_from_url(st.session_state.image_url, qr_path)
        
        # Final elapsed time
        final_time = time.time() - start_time
        timer_text.text(f"Total processing time: {final_time:.1f} seconds")
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Add a small delay to show the completion
        time.sleep(1)
        st.session_state.step = 'result'
        st.rerun()
        
    elif st.session_state.step == 'result':
        st.write("Your transformed photo is ready!")
        
        # Create two columns for the photo and QR code
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the final image
            image_path = f'final/image_{st.session_state.timestamp}.png'
            st.image(image_path, use_column_width=True)
        
        with col2:
            # Display QR code
            os.makedirs("qr", exist_ok=True)

            qr_path = f'qr/qr_code_{st.session_state.timestamp}.png'
            if os.path.exists(qr_path):
                st.image(qr_path, use_column_width=True)
            else:
                # Generate QR code if it doesn't exist
                generate_qr_from_url(st.session_state.image_url, qr_path)
                st.image(qr_path, use_column_width=True)
            
            st.write("Scan this QR code to download your photo!")
        
        # Email form below the images
        st.write("Enter your email to receive your photo:")
        email = st.text_input("Email Address")
        
        # Buttons in a single row, left-aligned
        if st.button("Send Photo"):
            if email:
                with st.spinner("Sending your photo..."):
                    send_email(email, st.session_state.image_url)
                    st.session_state.step = 'thank_you'
                    st.rerun()
            else:
                st.error("Please enter your email address")
        
        if st.button("Finish without sending"):
            # Clean up QR codes from the qr directory
            qr_path = f'qr/qr_code_{st.session_state.timestamp}.png'
            if os.path.exists(qr_path):
                try:
                    os.remove(qr_path)
                except Exception as e:
                    print(f"Could not remove {qr_path}: {e}")
            
            # Move to thank you step
            st.session_state.step = 'thank_you'
            st.rerun()

    elif st.session_state.step == 'thank_you':
        st.success("Thank you for using the AI Photobooth! Hope you enjoyed your experience!")
        st.balloons()
        if st.button("Start New Session"):
            # Clear all images and QR codes
        # Clean up QR codes from the qr directory
            if os.path.exists('qr'):
                for file in os.listdir('qr'):
                    if file.startswith('qr_code_') and file.endswith('.png'):
                        try:
                            os.remove(os.path.join('qr', file))
                        except Exception as e:
                            print(f"Could not remove {file}: {e}")
            
            # Reset session state
            st.session_state.step = 'start'
            st.session_state.timestamp = None
            st.session_state.image_url = None
            st.session_state.camera = None
            st.session_state.frame = None
            st.rerun()

if __name__ == "__main__":
    main() 