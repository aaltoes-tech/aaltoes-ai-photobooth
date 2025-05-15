# AI PhotoBooth

An application that transforms images of people by changing backgrounds and clothing styles while preserving faces and poses, using AI models.

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Segment Anything Model (SAM)
- FaceNet
- Replicate API access
- OpenAI API access
- Pinata API access for IPFS storage
- Resend API access for email functionality

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the SAM model checkpoint:
   ```
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```
   Place the downloaded model file in the root directory of the project.
   
4. Set up API keys in a `.env` file based on the provided `.env.example`:
   ```
   REPLICATE_API_TOKEN=your_replicate_api_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   PINATA_JWT=your_pinata_jwt_here
   RESEND_API_KEY=your_resend_api_key_here

   ```
   
   You'll need to:
   - Sign up for a [Replicate](https://replicate.com) account to get an API token
   - Sign up for an [OpenAI](https://openai.com) account to get an API key
   - Sign up for a [Pinata](https://pinata.cloud) account to get a JWT for IPFS storage
   - Sign up for a [Resend](https://resend.com) account to get an API key for email functionality

## Repository Structure

```
├── main.ipynb           # Main Jupyter notebook interface
├── pipeline.py          # Image processing pipeline coordinator
├── scripts.py           # Core functions for image processing
├── config.py            # Configuration settings
├── email_template.html  # HTML template for email delivery
├── requirements.txt     # Python dependencies
├── app.py               # Streamlit web interface
└── sam_vit_h_4b8939.pth # SAM model weights (download separately)
```

## How It Works

1. **Image Capture**: Captures an image from webcam or loads from file
2. **Face Detection**: Uses MTCNN (from FaceNet) to detect faces
3. **Face Mask Generation**: Uses Segment Anything Model (SAM) to create precise face masks
4. **Image Encoding**: Analyzes the image using OpenAI's GPT-4.1 to generate a description
5. **Image Transformation**: Uses Replicate API to run Ideogram v3 Turbo to transform clothing and background while preserving original faces
6. **Output Generation**: Creates a stacked comparison and polaroid-style image
7. **IPFS Storage**: Uploads the final image to IPFS via Pinata for permanent storage
8. **QR Code Generation**: Creates a QR code linking to the IPFS-stored image
9. **Email Delivery**: Optionally sends the result via email with the QR code

## Web UI Demo

The application includes a user-friendly web interface built with Streamlit. To run the web demo:

1. Make sure you have installed all dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run app.py`
3. Navigate to the URL shown in your terminal (typically http://localhost:8501)

The web interface allows you to:
- Upload an image or capture one using your webcam
- Customize clothing and background styles using text prompts
- Process the image with one click
- View the transformed result alongside a QR code for sharing
- Optionally send the result to your email

This interface is ideal for demonstrations and for users without technical experience.

## System Requirements

- For local processing: macOS with Apple Silicon (M1/M2/M3) or a machine with CUDA capability
- Internet connection for API access
- Approximately 16GB of RAM for efficient processing
- Disk space for temporary and output images

## Troubleshooting

- If you encounter memory issues, try reducing input image resolution
- If you get black output images, ensure SAM model is properly loaded
- For Mac users, the system uses MPS (Metal Performance Shaders) for hardware acceleration

## License

MIT License 