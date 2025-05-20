import os
import replicate
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2
from segment_anything import sam_model_registry, SamPredictor
from facenet_pytorch import MTCNN
import torch
import base64
import time
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import qrcode
import resend
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from config import replicate_model


load_dotenv()

REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINATA_JWT = os.environ.get("PINATA_JWT")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

def capture_image():
# Open webcam
    cap = cv2.VideoCapture()

    if not cap.isOpened():
        raise IOError("Cannot access webcam")

    time.sleep(2)

    # Discard a few initial frames (helps with camera exposure adjustment)
    for _ in range(10):
        cap.read()

    # Capture a frame
    ret, frame = cap.read()
    timestamp = time.time()

    # Create input directory if it doesn't exist
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)

    image_name = f"input/image_{timestamp}.jpg"

    if ret:
        cv2.imwrite(image_name, frame)
        print(f"Photo saved as {image_name}")
    else:
        print("Failed to capture image")

    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    return timestamp

def generate_face_mask(timestamp):
    image_name = f"input/image_{timestamp}.jpg"
    image = cv2.imread(image_name)
    
    if image is None:
        raise ValueError(f"Could not read image from {image_name}")
    
    # Resize the image to ensure dimensions are compatible with MPS
    h, w = image.shape[:2]
    # Ensure dimensions are multiples of 8, which works well with most ML models
    new_h, new_w = (h // 8) * 8, (w // 8) * 8
    if h != new_h or w != new_w:
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h} for MPS compatibility")
        image = cv2.resize(image, (new_w, new_h))
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load MTCNN face detector - always use CPU to avoid MPS issues
    print("Loading MTCNN face detector...")
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Detecting faces...")
    boxes, _, landmarks = mtcnn.detect(image_rgb, landmarks=True)
    
    if boxes is None or len(boxes) == 0:
        print("WARNING: No faces detected in the image!")
        # Create a blank mask if no faces are detected
        combined_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        os.makedirs("masks", exist_ok=True)
        cv2.imwrite(f"masks/image_{timestamp}.png", combined_mask)
        return
    
    print(f"Detected {len(boxes)} faces in the image")

    # Load SAM model
    print("Loading SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for SAM model")
    else:
        # Always use CPU on Mac to avoid MPS errors
        device = torch.device("mps")
        print("Using MPS for SAM model")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Prepare masks for all faces
    masks = []
    for i, (box, lm) in enumerate(zip(boxes, landmarks)):
        if box is None or lm is None:
            print(f"WARNING: Invalid box or landmarks for face {i}")
            continue
        x0, y0, x1, y1 = map(int, box)
        print(f"Processing face {i} at coordinates: ({x0}, {y0}, {x1}, {y1})")
        
        # 1) Prepare your prompts exactly as before:
        box_prompt = np.array([[x0, y0, x1, y1]])            # shape (1,4)
        eyes       = lm.astype(np.float32)               # shape (2,2)
        labels     = np.array([1]*len(eyes), dtype=np.int64)         # shape (2,)

        # 2) Ask SAM for mask:
        try:
            masks_for_face, _, _ = predictor.predict(
                point_coords=eyes,
                point_labels=labels,
                box=box_prompt,
                multimask_output=False
            )
            masks.append(masks_for_face[0])
            print(f"Successfully generated mask for face {i}")
        except Exception as e:
            print(f"Error generating mask for face {i}: {str(e)}")
            continue

    if not masks:
        print("WARNING: No valid masks were generated!")
        # Create a blank mask if no valid masks were generated
        combined_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    else:
        # Combine all masks into one
        combined_mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
        print(f"Successfully combined {len(masks)} masks")

    # Save output
    os.makedirs("masks", exist_ok=True)
    mask_path = f"masks/image_{timestamp}.png"
    cv2.imwrite(mask_path, combined_mask)
    print(f"Saved mask to {mask_path}")

def generate_encoding_image(prompt_text, timestamp):
    client = OpenAI(api_key=OPENAI_API_KEY)
    image_path = f"input/image_{timestamp}.jpg"
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    # build the messages list with proper content types
    message = {
        "role": "user",
        "content": [
            { "type": "text",       "text": prompt_text },
            { 
              "type": "image_url",  
              "image_url": { 
                  "url": f"data:image/jpeg;base64,{b64_data}" 
              } 
            }
        ],
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",      # or your multimodal-enabled model
        messages=[message],
    )

    return resp.choices[0].message.content

def generate_polaroid_image(encoding_image, clothes_prompt, background_prompt, reference_name, timestamp, model="ideogram-ai/ideogram-v3-turbo"):
    """Generate a polaroid image with the selected style.
    
    Args:
        encoding_image (str): The image encoding text
        clothes_prompt (str): The clothes prompt text (deprecated, kept for backward compatibility)
        background_prompt (str): The background prompt text (deprecated, kept for backward compatibility)
        timestamp (float): The timestamp for file naming
        model (str): The model to use for generation
        style_index (int): Index of the style to use (0-3)
    """
    # Use the selected style prompts from config
    prompt = f"{encoding_image} \n{background_prompt} \n{clothes_prompt}"

    # initialize the client with your API key
    client = replicate.Client(api_token=REPLICATE_API_KEY)

    # Upload local files
    photo_path = f"input/image_{timestamp}.jpg"
    mask_path = f"masks/image_{timestamp}.png"

    # Verify files exist
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo not found at {photo_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    # Verify mask is not empty
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    
    # Check if mask is empty (all zeros)
    if np.all(mask_img == 0):
        print("WARNING: Mask is empty (all zeros)! This means no faces were detected or mask generation failed.")
    else:
        # Calculate percentage of non-zero pixels
        mask_percentage = (np.count_nonzero(mask_img) / mask_img.size) * 100
        print(f"Mask contains {mask_percentage:.2f}% non-zero pixels")

    print(f"Opening photo from {photo_path}")
    image = open(photo_path, "rb")
    print(f"Opening mask from {mask_path}")
    mask = open(mask_path, "rb")

    reference = open(reference_name, "rb")

    print("Running Ideogram model with image and mask...")
    # Run the model
    output = client.run(
        model,  # Use the provided model parameter
        input={
            "image": image,                # Use uploaded URL
            "mask": mask,
            "prompt": prompt,
            "resolution": "None",
            "style_type": "Realistic",
            "aspect_ratio": "3:2",
            "magic_prompt_option": "Off",
            "style_reference_images": [reference]
        }
    )
    print("Model run completed")

    resp = requests.get(output)
    resp.raise_for_status()

    os.makedirs("outputs", exist_ok=True)

    filename = f"outputs/image_{timestamp}.png"
    with open(filename, "wb") as f:
        f.write(resp.content)

    def add_rounded_corners(image, radius=30):
        """Add rounded corners to an image"""
        # Create a mask with rounded corners
        mask = Image.new('L', image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
        
        # Create a new image with transparency
        output = Image.new('RGBA', image.size, (255, 255, 255, 0))
        
        # Convert image to RGBA if it isn't already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Paste the image onto the output using the mask
        output.paste(image, (0, 0), mask)
        return output

    def stack_images_vertically(top_path, bottom_path, 
                                output_path, spacing=20, corner_radius=30, border_width=10, border_color=(80, 80, 80)):
        # Open images
        img_top = Image.open(top_path)
        img_bot = Image.open(bottom_path)

        # Determine target width
        max_width = max(img_top.width, img_bot.width)

        # Resize narrower image(s) to match max_width, preserving aspect ratio
        def resize_to_width(img, target_width):
            if img.width == target_width:
                return img
            # compute new height to preserve aspect ratio
            new_height = int(img.height * (target_width / img.width))
            return img.resize((target_width, new_height), Image.LANCZOS)

        img_top = resize_to_width(img_top, max_width)
        img_bot = resize_to_width(img_bot, max_width)

        def add_border_and_round_corners(img, border_width, border_color, corner_radius):
            # Create a new image with border
            border_img = Image.new('RGB', 
                (img.width + 2*border_width, img.height + 2*border_width), 
                border_color)
            
            # Create a mask for rounded corners
            mask = Image.new('L', border_img.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle(
                [(0, 0), border_img.size], 
                radius=corner_radius, 
                fill=255
            )
            
            # Create a new image with transparency
            result = Image.new('RGBA', border_img.size, (0, 0, 0, 0))
            
            # Convert border image to RGBA
            border_img = border_img.convert('RGBA')
            
            # Paste the border image using the mask
            result.paste(border_img, (0, 0), mask)
            
            # Convert original image to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create a mask for the inner image
            inner_mask = Image.new('L', img.size, 0)
            inner_mask_draw = ImageDraw.Draw(inner_mask)
            inner_mask_draw.rounded_rectangle(
                [(0, 0), img.size], 
                radius=corner_radius - border_width, 
                fill=255
            )
            
            # Paste the original image on top of the border
            result.paste(img, (border_width, border_width), inner_mask)
            
            return result

        # Add borders and round corners to both images
        img_top = add_border_and_round_corners(img_top, border_width, border_color, corner_radius)
        img_bot = add_border_and_round_corners(img_bot, border_width, border_color, corner_radius)

        # Create a new image with combined height plus spacing and black background
        total_height = img_top.height + img_bot.height + spacing
        new_img = Image.new('RGB', (max_width + 2*border_width, total_height), (0, 0, 0))

        # Paste images with spacing between them
        new_img.paste(img_top, (0, 0), img_top)
        new_img.paste(img_bot, (0, img_top.height + spacing), img_bot)

        # Save the result
        os.makedirs("stacked", exist_ok=True)
        new_img.save(output_path)
        print(f"Saved stacked image with gray borders to {output_path}")

    # Example usage with spacing and rounded corners:
    stack_images_vertically(
        top_path=f"input/image_{timestamp}.jpg",
        bottom_path=f"outputs/image_{timestamp}.png",
        output_path=f"stacked/image_{timestamp}.jpg",
        spacing=40,
        corner_radius=50  # Added corner radius parameter
    )

    def make_polaroid(
        photo_path: str,
        output_path: str,
        photo_width: int = 800,
        border: int = 90,
        bottom_border: int = 150,
        bg_color = (0, 0, 0),  # Black background
        logo_path: str = "aaltoes_white.png",  # Default to white logo
        logo_margin_top: int = 50,
    ):
        # 1) Load & resize the photo
        img = Image.open(photo_path)
        ratio = photo_width / img.width
        target_size = (photo_width, int(img.height * ratio))
        img = img.resize(target_size, Image.LANCZOS)

        # 2) Create canvas with black background
        canvas_w = target_size[0] + 2 * border
        canvas_h = target_size[1] + border + bottom_border
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        canvas.paste(img, (border, border))

        # 3) Add logo at original size
        if logo_path and os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")
            logo_x = (canvas_w - logo.width) // 2
            logo_y = target_size[1] + border + logo_margin_top
            canvas.paste(logo, (logo_x, logo_y), logo)

        # 4) Save
        os.makedirs("final", exist_ok=True)
        canvas.save(output_path)
        print(f"Polaroid with logo saved to {output_path}")

    make_polaroid(
        photo_path=f"stacked/image_{timestamp}.jpg",
        output_path=f"final/image_{timestamp}.png",
        photo_width=1080+2*45,
        border= 45,
        bottom_border=300,
        bg_color=(0, 0, 0),
        logo_path="aaltoes_white.png",
        logo_margin_top=90  # More space above the logo
    )

def show_image(timestamp, folder, ext = "jpg"):

    root_dir = os.getcwd()
    image_path = os.path.join(root_dir, folder, f"image_{timestamp}.{ext}")

    img = Image.open(image_path)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def upload_file(file_path: str):
    """Upload a file to Pinata"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if not PINATA_JWT:
        raise ValueError("PINATA_JWT environment variable is not set")
        
    # Prepare the file
    with open(file_path, 'rb') as file:
        # Prepare multipart form data
        files = {
            'file': (os.path.basename(file_path), file),
            'pinataMetadata': (None, '{"name": "' + os.path.basename(file_path) + '"}'),
            'pinataOptions': (None, '{"cidVersion": 1}')
        }
            
        # Make the request
        headers = {"Authorization": f"Bearer {PINATA_JWT}"}
        response = requests.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            files=files,
            headers=headers
        )
        
        if response.status_code == 200:
            result = f"https://amaranth-defiant-snail-192.mypinata.cloud/ipfs/{response.json()['IpfsHash']}"
            return result
        else:
            raise Exception(f"Failed to upload file: {response.text}")

def generate_qr_from_url(url: str, output_path: str = "qr_code.png") -> Image.Image:
    """
    Generates a QR code for the given URL and saves it to output_path.
    Returns the PIL Image object.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    return img

def clear_images(timestamp, keep_final=False):
    """
    Clean temporary image files for a specific timestamp.
    
    Args:
        timestamp (str): The timestamp of the images to clean
        keep_final (bool, optional): Whether to keep the final image. Defaults to False.
    """
    # List of files to remove
    files_to_remove = [
        f"input/image_{timestamp}.jpg",
        f"masks/image_{timestamp}.png",
        f"outputs/image_{timestamp}.png",
        f"stacked/image_{timestamp}.jpg"
    ]
     
    # Track which files were removed
    removed_count = 0
    
    # Safely remove each file if it exists
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    print(f"Cleaned {removed_count} temporary images")

def clean_all_images(current_timestamp=None):
    """
    Clean all images from input and final directories except for the current_timestamp if provided.
    
    Args:
        current_timestamp (str, optional): Timestamp of the current session to preserve
    """
    directories = ["input", "final", "masks", "outputs", "stacked"]
    
    total_removed = 0
    
    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Clean each directory
    for directory in directories:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip if it's a directory
            if os.path.isdir(file_path):
                continue
                
            # Only remove files with timestamp in name
            if "image_" in filename:
                # If we have a current timestamp, only remove other files
                if current_timestamp and f"image_{current_timestamp}" in filename:
                    continue
                    
                try:
                    os.remove(file_path)
                    total_removed += 1
                except Exception as e:
                    print(f"Could not remove {file_path}: {e}")
    
    print(f"Cleaned {total_removed} image files from all directories")

def send_email(to_email, image_url):
    """
    Send an email with the AI PhotoBooth image and QR code
    
    Args:
        to_email (str): Recipient email address
        image_url (str): URL of the generated image
    
    Returns:
        dict: The response from Resend API
    """
    
    # Set the API key
    resend.api_key = os.environ.get("RESEND_API_KEY")
    if not resend.api_key:
        raise ValueError("RESEND_API_KEY not set in environment variables")
    
    # Load the email template
    try:
        with open("email_template.html", "r") as file:
            template = file.read()
            
        # Replace placeholders with actual values
        html_content = template.replace("{{image_url}}", image_url)
        
        # Download the image from the URL to attach it
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            attachments = [
                {
                    "filename": "your_transformed_photo.png",
                    "content": base64.b64encode(image_data).decode("utf-8")
                }
            ]
        else:
            print(f"Warning: Could not download image from URL for attachment. Status code: {response.status_code}")
            attachments = []
        
        # Prepare email parameters
        params = {
            "from": "AI PhotoBooth <photobooth@aaltoes.com>",
            "to": to_email,
            "subject": "Your Unsung Heroes Photo",
            "html": html_content,
            "attachments": attachments
        }
        
        # Send email using the resend library
        try:
            response = resend.Emails.send(params)
            print(f"Email sent to {to_email}")
            return response
        except AttributeError:
            # If Emails.send doesn't work, try emails.send (depends on resend version)
            response = resend.emails.send(params)
            print(f"Email sent to {to_email}")
            return response
            
    except Exception as e:
        print(f"Error sending email: {e}")
        return {"error": str(e)}
    


def generate_pdf(timestamp):
    # ─── 1) CONFIGURE THESE PATHS ────────────────────────────────────
    main_image_path = f'final/image_{timestamp}.png'  
    qr_image_path   = 'qr_code.png'      # ← your right-column QR code
    logo_path       = 'aaltoes_white.png'         # ← bottom-center logo
    output_pdf      = 'booth_thanks_horizontal.pdf'
    # ────────────────────────────────────────────────────────────────

    # Create a landscape A4 PDF
    c = canvas.Canvas(output_pdf, pagesize=landscape(A4))
    page_w, page_h = landscape(A4)

    # ─── 1) DRAW LOGO AT THE VERY TOP ─────────────────────────────
    with Image.open(logo_path) as logo:
        lw, lh = logo.size
    logo_max_w = page_w / 6
    logo_scale = min(logo_max_w / lw, 1.0)
    lw_d, lh_d = lw * logo_scale, lh * logo_scale
    logo_x = (page_w - lw_d) / 2
    logo_y = page_h - lh_d - 20
    c.drawImage(logo_path, logo_x, logo_y, width=lw_d, height=lh_d, mask='auto')

    # ─── 2) DRAW THANK-YOU TEXT BELOW THE LOGO ────────────────────
    text = "Thank you for attending our booth"
    c.setFont("Helvetica", 24)
    text_w = c.stringWidth(text, "Helvetica", 24)
    text_x = (page_w - text_w) / 2
    text_y = logo_y - 50
    c.drawString(text_x, text_y, text)

    # ─── 3) TWO‐COLUMN LAYOUT FOR MAIN IMAGE + QR ────────────────
    margin        = 50
    gutter        = 20
    usable_w      = page_w - 2*margin - gutter
    col_w         = usable_w / 2
    top_border    = text_y - 40
    bottom_border = 100   # leave space for potential footer

    # -- Left column: main image
    with Image.open(main_image_path) as img:
        iw, ih = img.size
    scale = min(col_w/iw, (top_border - bottom_border)/ih, 1.0)
    draw_w, draw_h = iw*scale, ih*scale
    img_x = margin + (col_w - draw_w)/2
    img_y = bottom_border + ((top_border - bottom_border) - draw_h)/2
    c.drawImage(main_image_path, img_x, img_y, width=draw_w, height=draw_h)

    # -- Right column: caption above QR code
    qr_size = 200

    # Calculate the Y‐position of the QR so we can stack the caption on top
    qr_x = margin + col_w + gutter + (col_w - qr_size) / 2
    qr_y = bottom_border + ((top_border - bottom_border) - qr_size) / 2

    # 1) Draw caption above the QR
    caption = "Scan to download image"
    c.setFont("Helvetica", 12)
    cap_w = c.stringWidth(caption, "Helvetica", 12)
    cap_x = qr_x + (qr_size - cap_w) / 2
    cap_y = qr_y + qr_size + 8   # 8pt gap above the QR
    c.drawString(cap_x, cap_y, caption)

    # 2) Now draw the QR code below it
    c.drawImage(qr_image_path, qr_x, qr_y, width=qr_size, height=qr_size)

    # ─── 4) SAVE PDF ───────────────────────────────────────────────
    c.save()
    print(f"✅ Generated: {output_pdf}")