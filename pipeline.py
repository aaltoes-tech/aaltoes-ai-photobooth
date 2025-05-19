from scripts import *
from config import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--email", type=str, default="milka3341@gmail.com", help="Email address to send the transformed image to")
args = parser.parse_args()

if __name__ == "__main__":
    
    email = args.email
    timestamp = capture_image()
    encoding_image = generate_encoding_image(prompt_text, timestamp)
    generate_face_mask(timestamp)
    generate_polaroid_image(encoding_image, clothes_prompt, background_prompt,timestamp)
    url = upload_file(f"final/image_{timestamp}.png")

    generate_pdf(timestamp)
    clear_images(timestamp)

    send_email(args.email, url)