# Aaltoes AI Photobooth

An AI-powered photobooth that transforms photos into unique artistic styles using AI image generation.

## Features

- **Multiple AI Styles**: Choose from four unique styles:
  - Bioluminescent Explorer
  - Workshop Inventor
  - Cosmic Scholar
  - Nocturnal Scientist
- **Multiple AI Models**: Select from different Ideogram models based on your needs:
  - Turbo: Fastest processing with good quality (Recommended for quick results)
  - Balanced: Balanced speed and quality (Good for most use cases)
  - Quality: Highest quality but slower processing (Best for final results)
- **Photo Input Options**:
  - Webcam capture
  - Photo upload
- **Face Detection**: Automatically detects faces in photos for optimal transformation
- **Email Delivery**: Get your transformed photos delivered to your email
- **QR Code**: Easy download access via QR code

## Requirements

- Python 3.8 or higher
- Webcam (for live capture)
- Required Python packages (see `requirements.txt`)
- Environment variables:
  - `REPLICATE_API_KEY`: For AI image generation
  - `OPENAI_API_KEY`: For image analysis
  - `PINATA_JWT`: For file storage
  - `RESEND_API_KEY`: For email delivery

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aaltoes-ai-photobooth.git
cd aaltoes-ai-photobooth
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Download required models:
```bash
# Download SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Choose your preferred style and AI model

3. Take a photo using webcam or upload an existing photo

4. Preview your photo and continue to processing

5. Wait for the AI transformation (typically 30-40 seconds)

6. Get your transformed photo:
   - Scan the QR code to download
   - Enter your email to receive it

## Photo Requirements

- Clear, well-lit photos
- Supported formats: JPG, JPEG, PNG
- Recommended resolution: 1080x720 or higher
- For best results:
  - Ensure faces are clearly visible
  - Use good lighting
  - Avoid blurry images

## File Handling

- Input photos are automatically resized to 1080x720
- Original orientation is preserved
- Face masks are generated for proper AI transformation
- Final images are saved in "polaroid" template.

## Error Handling

- Face Detection: If no faces are detected, you'll be prompted to try a different photo
- Webcam Access: Clear error messages if webcam is unavailable
- File Upload: Automatic handling of file formats and orientations
- Processing: Progress indicators and status updates during transformation

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ideogram AI for image generation
- OpenAI for image analysis
- Segment Anything Model (SAM) for face masking
- Aaltoes for the project support 