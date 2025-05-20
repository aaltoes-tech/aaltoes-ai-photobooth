import streamlit as st
import os
import time
from PIL import Image
from scripts import capture_image, generate_face_mask, generate_encoding_image, generate_polaroid_image, show_image, upload_file, generate_qr_from_url, send_email, clear_images, clean_all_images
from config import prompt_text, clothes_prompt, background_prompt, replicate_model

def initialize_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 'start'
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = None
    if 'photo_source' not in st.session_state:
        st.session_state.photo_source = None
    if 'selected_style' not in st.session_state:
        st.session_state.selected_style = 0  # Default to first style

def main():
    initialize_session_state()
    
    # Style selection at the start
    if st.session_state.step == 'start':
        st.title("AI PhotoBooth")
        
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
        
        # Capture method selection
        capture_method = st.radio(
            "Choose how to capture your photo:",
            ["Webcam", "Upload Photo"],
            horizontal=True
        )
        
        if capture_method == "Webcam":
            st.session_state.photo_source = 'webcam'
            if st.button("Start Camera"):
                try:
                    timestamp = capture_image()
                    st.session_state.timestamp = timestamp
                    st.session_state.step = 'preview'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error capturing image: {str(e)}")
        else:
            st.session_state.photo_source = 'upload'
            uploaded_file = st.file_uploader("Upload a photo", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    # Create input directory if it doesn't exist
                    os.makedirs("input", exist_ok=True)
                    
                    # Save the uploaded file with timestamp
                    timestamp = time.time()
                    image_path = f"input/image_{timestamp}.jpg"
                    
                    # Convert and save the image
                    image = Image.open(uploaded_file)
                    image = image.convert('RGB')
                    image.save(image_path)
                    
                    st.session_state.timestamp = timestamp
                    st.session_state.step = 'preview'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    elif st.session_state.step == 'preview':
        st.title("Preview")
        
        # Display the captured/uploaded image
        image_path = f"input/image_{st.session_state.timestamp}.jpg"
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Upload Different Photo" if st.session_state.photo_source == 'upload' else "Retake Photo"):
                    st.session_state.step = 'start'
                    st.rerun()
            with col2:
                if st.button("Continue"):
                    st.session_state.step = 'processing'
                    st.rerun()
        else:
            st.error("Image not found. Please try again.")
            st.session_state.step = 'start'
            st.rerun()

    elif st.session_state.step == 'processing':
        st.title("Processing")
        
        with st.spinner("Generating your AI-enhanced photo..."):
            try:
                # Generate face mask
                generate_face_mask(st.session_state.timestamp)
                
                # Generate encoding image
                encoding_image = generate_encoding_image(prompt_text, st.session_state.timestamp)
                
                # Generate polaroid image with selected style
                generate_polaroid_image(
                    encoding_image, 
                    st.session_state.timestamp, 
                    replicate_model,
                    style_index=st.session_state.selected_style
                )
                
                st.session_state.step = 'result'
                st.rerun()
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.session_state.step = 'start'
                st.rerun()

    elif st.session_state.step == 'result':
        st.title("Your AI-Enhanced Photo")
        
        # Display the final image
        final_image_path = f"final/image_{st.session_state.timestamp}.png"
        if os.path.exists(final_image_path):
            st.image(final_image_path, use_column_width=True)
            
            # Upload to IPFS and generate QR code
            try:
                image_url = upload_file(final_image_path)
                qr_path = f"qr_code_{st.session_state.timestamp}.png"
                generate_qr_from_url(image_url, qr_path)
                
                # Display QR code
                st.image(qr_path, caption="Scan to download your photo", width=200)
                
                # Email form
                with st.form("email_form"):
                    email = st.text_input("Enter your email to receive the photo:")
                    if st.form_submit_button("Send Email"):
                        if email:
                            try:
                                send_email(email, image_url)
                                st.success("Email sent successfully!")
                            except Exception as e:
                                st.error(f"Error sending email: {str(e)}")
                        else:
                            st.warning("Please enter an email address")
                
                # Clean up option
                if st.button("Start New Session"):
                    clean_all_images(st.session_state.timestamp)
                    st.session_state.step = 'start'
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating QR code: {str(e)}")
        else:
            st.error("Final image not found. Please try again.")
            st.session_state.step = 'start'
            st.rerun()

if __name__ == "__main__":
    main() 