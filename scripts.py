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


load_dotenv()

REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINATA_JWT = os.environ.get("PINATA_JWT")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

def capture_image():
# Open webcam
    cap = cv2.VideoCapture(0)

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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    boxes, _, landmarks = mtcnn.detect(image_rgb, landmarks=True)

    # Load SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS on Mac (Apple Silicon)")
    else:
        device = torch.device("cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Prepare masks for all faces
    masks = []
    for box, lm in zip(boxes, landmarks):
        if box is None or lm is None:
            continue
        x0, y0, x1, y1 = map(int, box)
    # 1) Prepare your prompts exactly as before:
        box_prompt = np.array([[x0, y0, x1, y1]])            # shape (1,4)
        eyes       = lm.astype(np.float32)               # shape (2,2)
        labels     = np.array([1]*len(eyes), dtype=np.int64)         # shape (2,)

        # 2) Ask SAM for three masks:
        masks_for_face, _, _ = predictor.predict(
            point_coords=eyes,
            point_labels=labels,
            box=box_prompt,
            multimask_output=False    # ← here's the change
        )

        masks.append(masks_for_face[0])

    # Combine all masks into one
    combined_mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
    # Save output
    os.makedirs("masks", exist_ok=True)

    cv2.imwrite(f"masks/image_{timestamp}.png", combined_mask)

def generate_encoding_image(prompt_text, timestamp):

    image_name = f"input/image_{timestamp}.jpg"
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(image_name, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt_text },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                    },
                ],
            }
        ],
    )

    encoding_image = response.output_text

    return encoding_image


def generate_polaroid_image(encoding_image, clothes_prompt, background_prompt,timestamp):

    prompt = f"{encoding_image} \n{clothes_prompt} \n{background_prompt}"

    # initialize the client with your API key
    client = replicate.Client(api_token=REPLICATE_API_KEY)

    # Upload local files
    photo_path = f"input/image_{timestamp}.jpg"
    mask_path = f"masks/image_{timestamp}.png"

    image = open(photo_path, "rb")
    mask = open(mask_path, "rb")

    # Run the model
    output = client.run(
        "ideogram-ai/ideogram-v3-turbo",
        input={
            "image": image,                # Use uploaded URL
            "mask": mask,
            "prompt": prompt,
            "resolution": "None",
            "style_type": "None",
            "aspect_ratio": "3:2",
            "magic_prompt_option": "Off"
        }
    )

    resp = requests.get(output)
    resp.raise_for_status()

    os.makedirs("outputs", exist_ok=True)

    filename = f"outputs/image_{timestamp}.png"
    with open(filename, "wb") as f:
        f.write(resp.content)

    def stack_images_vertically(top_path, bottom_path, output_path):
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

        # Create a new image with combined height
        total_height = img_top.height + img_bot.height
        new_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))

        # Paste images
        new_img.paste(img_top, (0, 0))
        new_img.paste(img_bot, (0, img_top.height))

        # Save the result
        os.makedirs("stacked", exist_ok=True)
        new_img.save(output_path)
        print(f"Saved stacked image to {output_path}")


        # Example usage:
    stack_images_vertically(
        top_path=f"input/image_{timestamp}.jpg",
        bottom_path=f"outputs/image_{timestamp}.png",
        output_path=f"stacked/image_{timestamp}.jpg"
    )


    def make_polaroid(
        photo_path: str,
        output_path: str,
        caption: str,
        photo_width: int = 600,
        border: int = 20,
        bottom_border: int = 120,
        bg_color = (255,255,255),
        font_path: str = None,
        font_size: int = 28,
        logo_path: str = None,
        logo_size: tuple[int,int] = None,
        logo_margin_top: int = 10,
        caption_margin_top: int = 10,
    ):
        # 1) Load & resize the photo
        img = Image.open(photo_path)
        ratio = photo_width / img.width
        target_size = (photo_width, int(img.height * ratio))
        img = img.resize(target_size, Image.LANCZOS)

        # 2) Create canvas
        canvas_w = target_size[0] + 2 * border
        canvas_h = target_size[1] + border + bottom_border
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        canvas.paste(img, (border, border))

        # 3) Prepare to draw
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        # 4) Paste & center the logo
        logo_y0 = target_size[1] + border + logo_margin_top
        if logo_path:
            logo = Image.open(logo_path).convert("RGBA")
            if logo_size:
                logo = logo.resize(logo_size, Image.LANCZOS)
            logo_x = (canvas_w - logo.width) // 2
            canvas.paste(logo, (logo_x, logo_y0), logo)

        # 5) Measure caption
        try:
            bbox = draw.textbbox((0,0), caption, font=font)
            text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except AttributeError:
            mask = font.getmask(caption)
            text_w, text_h = mask.size

        # 6) Position caption below logo, centered
        text_x = (canvas_w - text_w) // 2
        # If there was a logo, stack below it; otherwise center in bottom border
        if logo_path:
            text_y = logo_y0 + logo.height + caption_margin_top
        else:
            # center vertically in bottom border
            bottom_y0 = target_size[1] + border
            text_y = bottom_y0 + (bottom_border - text_h) // 2

        draw.text((text_x, text_y), caption, fill=(0,0,0), font=font)

        # 7) Save
        os.makedirs("final", exist_ok=True)
        canvas.save(output_path)
        print(f"Polaroid with centered logo & caption saved to {output_path}")

    make_polaroid(
        photo_path=f"stacked/image_{timestamp}.jpg",
        output_path=f"final/image_{timestamp}.png",
        caption="Unsung heroes",
        photo_width=1000,
        border=30,
        bottom_border=160,
        font_path="/Library/Fonts/Geist-Regular.ttf",  # or wherever your .ttf is
        font_size=32,
        logo_path="aaltoes_dark.png",      # your transparent PNG
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

def clear_images(timestamp):
    # Remove temp files
    os.remove(f"input/image_{timestamp}.jpg")
    os.remove(f"masks/image_{timestamp}.png")
    os.remove(f"outputs/image_{timestamp}.png")
    os.remove(f"stacked/image_{timestamp}.jpg")
    os.remove(f"final/image_{timestamp}.png")
    print("Temporary images cleared")

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
    logo_path       = 'aaltoes_dark.png'         # ← bottom-center logo
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