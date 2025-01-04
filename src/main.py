import os
import re
import csv
import base64
import xml.etree.ElementTree as ET

import cv2
import requests
from PIL import Image
from ultralytics import YOLO

# Function to read and extract data from the CSV file
def read_csv(file_path):
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)  # Read CSV as a dictionary
            data = [row for row in csv_reader]
            
        return data
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except KeyError as e:
        print(f"Error: Missing column {e} in the CSV file.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def download_image_google_drive(image_url):
    id = image_url.split('=')[-1]
    
    if not id or len(id) < 2:
        id = image_url.split('/')[-2]
    
    downloadable_url = f"https://drive.google.com/uc?export=view&id={id}"
    response = requests.get(downloadable_url)
    
    with open(f'temp.png', 'wb') as file:
        file.write(response.content)
        
    return f'temp.png'

def get_face_center_image(image_path):
    # image = cv2.imread(image_path)
    # org_width, org_height = image.shape[1], image.shape[0]

    # # Detect faces in the image
    # faces = detector.detect_faces(image)

    # if not faces:
    #     print("No face detected!")
    # else:
    #     # Loop through each detected face
    #     for face in faces:
    #         # Extract the bounding box of the face
    #         x1, y1, width, height = face['box']
    #         x2, y2 = x1 + width, y1 + height

    #         # Crop the image to the detected face
    #         cropped_image = image[y1:y2, x1:x2]

    #         # Convert the cropped image from BGR to RGB (for PIL)
    #         cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
    #         # Convert to Pillow Image format for further processing
    #         pil_image = Image.fromarray(cropped_image_rgb)

    #         # Calculate the dimensions for the final square image
    #         width, height = pil_image.size
    #         square_size = max(width, height)  # Use the larger dimension as the new square size

    #         # Create a new square image with a white background
    #         new_image = Image.new('RGB', (square_size, square_size), (255, 255, 255))

    #         # Calculate the position to center the cropped face
    #         position = ((square_size - width) // 2, (square_size - height) // 2)

    #         # Paste the cropped face into the center of the new square image
    #         new_image.paste(pil_image, position)

    #         # Save or display the final centered face image
    #         new_image.save(image_path)  # or any desired output path
    #         print(f"Face cropped and centered, saved as {image_path}")
    
    # Load the pre-trained HOG + SVM detector for detecting people


    # Read the image


    # Load YOLO
    # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    # layer_names = net.getLayerNames()
    # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # # Load image
    # image = cv2.imread(image_path)
    # height, width, channels = image.shape

    # # Prepare image for YOLO
    # blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)

    # # Loop over all detections
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]

    #         if confidence > 0.5 and class_id == 0:  # Class ID for person is 0 in YOLO
    #             center_x = int(detection[0] * width)
    #             center_y = int(detection[1] * height)
    #             w = int(detection[2] * width)
    #             h = int(detection[3] * height)

    #             # Get the bounding box coordinates
    #             x = center_x - w // 2
    #             y = center_y - h // 2

    #             # Crop the image around the detected person
    #             cropped_image = image[y:y+h, x:x+w]

    #             # Convert to Pillow Image format for further processing
    #             pil_image = Image.fromarray(cropped_image)

    #             # Save or process the cropped image (center it or add padding if needed)
    #             pil_image.save(image_path)
    #             print(f"Person cropped and saved as {image_path}")


    return image_path

def center_person_in_image(image_path):
    # Load YOLO model (YOLOv8 pretrained model)
    model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is a pre-trained lightweight YOLO model

    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Run inference
    results = model(image)

    # Extract detections for 'person' class (class ID for 'person' in COCO is 0)
    person_boxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box
            if int(class_id) == 0 and confidence > 0.5:  # Person class with confidence > 50%
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    if not person_boxes:
        print("No person detected in the image!")
        return None

    # Get the largest detected person
    x1, y1, x2, y2 = max(person_boxes, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))

    # Calculate the center and define a square crop
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    square_size = max(x2 - x1, y2 - y1) * 1.5  # Slightly larger square crop

    # Define square boundaries
    x_start = max(int(center_x - square_size / 2), 0)
    y_start = max(int(center_y - square_size / 2), 0)
    x_end = min(int(center_x + square_size / 2), width)
    y_end = min(int(center_y + square_size / 2), height)

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    final_image = cv2.resize(cropped_image, (600, 600))

    cv2.imwrite(image_path, final_image)
    print(f"Centered image saved as {image_path}")

    return image_path

def standardize_id_image(image_path, output_path):
    # Load YOLO model
    model = YOLO('yolov8n.pt')

    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Detect persons
    results = model(image)
    person_boxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box
            if int(class_id) == 0 and confidence > 0.5:  # Class 0: Person
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    if not person_boxes:
        print("No person detected!")
        return None
    
    # Use the largest detected person
    x1, y1, x2, y2 = max(person_boxes, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))

    # Crop the detected person
    cropped_image = image[y1:y2, x1:x2]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil_image = Image.fromarray(cropped_image)

    # Ensure image is a consistent aspect ratio (e.g., 4:5)
    target_aspect_ratio = 4 / 5
    current_aspect_ratio = pil_image.width / pil_image.height

    if current_aspect_ratio > target_aspect_ratio:
        # Too wide, crop horizontally
        new_width = int(pil_image.height * target_aspect_ratio)
        offset = (pil_image.width - new_width) // 2
        pil_image = pil_image.crop((offset, 0, offset + new_width, pil_image.height))
    else:
        # Too tall, crop vertically
        new_height = int(pil_image.width / target_aspect_ratio)
        offset = (pil_image.height - new_height) // 2
        pil_image = pil_image.crop((0, offset, pil_image.width, offset + new_height))
    
    # Resize to standard size (300x400)
    pil_image = pil_image.resize((300, 400), Image.LANCZOS)

    # Optional: Add a white background
    final_image = Image.new('RGB', (300, 400), (255, 255, 255))
    final_image.paste(pil_image, (0, 0))

    # Save the standardized image
    final_image.save(output_path)
    print(f"Standardized image saved at {output_path}")
    return output_path

def get_image(image_url):
    image_path = download_image_google_drive(image_url)
    
    try:
        image_path = center_person_in_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Using the original image without centering the person.")
        
    return image_path

def remove_svg_namespace(svg_content):
    # Remove namespace prefixes (e.g., ns0:)
    svg_content = re.sub(r'\s?xmlns:ns0="[^"]+"', '', svg_content)  # Remove xmlns declaration
    svg_content = re.sub(r'ns0:', '', svg_content)  # Remove ns0 prefix
    svg_content = re.sub(r'ns1:', '', svg_content)  # Remove ns1 prefix if present
    return svg_content

def generate_id(svg_template, output_file, name, email, user_id, bg, image_path):
    # Parse the SVG template
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ET.parse(svg_template)
    root = tree.getroot()
    
    # Define the namespace for XPath queries
    namespace = {'svg': 'http://www.w3.org/2000/svg'}

    # Update text fields
    name_elem = root.find(".//svg:text[@id='name']", namespace)
    if name_elem is not None:
        if len(name) > 20:
            name_elem.set("font-size", "11px")
        name_elem.text = name

    email_elem = root.find(".//svg:text[@id='email']", namespace)
    if email_elem is not None:
        if len(email) > 30:
            email_elem.set("font-size", "10px")
        email_elem.text = email

    id_elem = root.find(".//svg:text[@id='id']", namespace)
    if id_elem is not None:
        id_elem.text = user_id

    bg_elem = root.find(".//svg:text[@id='bg']", namespace)
    if bg_elem is not None:
        bg_elem.text = bg

    # Update image field
    image_elem = root.find(".//svg:image[@id='image']", namespace)
    if image_elem is not None:
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")
        image_elem.set(f"{{{namespace['svg']}}}href", f"data:image/{image_path.split('.')[-1]};base64,{img_data}")
    
    # Save the updated SVG
    tree.write(output_file)
    
    with open(output_file, 'r') as file:
        svg_content = file.read()

    clean_svg = remove_svg_namespace(svg_content)

    with open(output_file, 'w') as file:
        file.write(clean_svg)
        
    print(f"SVG updated and saved to {output_file}") 

def svg_to_pdf(svg_file, pdf_file):
    try:
        # Convert SVG to PDF
        os.system(f"rsvg-convert -f pdf -o {pdf_file} {svg_file}")
        print(f"PDF saved as {pdf_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_mode = True
    test_size = 5
    
    CSV_FILE = 'data/data.csv'
    TEMPLATE_FILE = "templates/card.svg"
    OUTPUT_FOLDER = "output"
    OUTPUT_PDF = "output/pdfs"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_PDF, exist_ok=True)

    extracted_data = read_csv(CSV_FILE)
    if extracted_data:
        print("Extracted Data from CSV:\n")
        try:
            for idx, entry in enumerate(extracted_data):
                if "none" not in entry.values():
                    print(f"Working on entry: {entry['ID']}\n")
                    image_path = get_image(entry['Photo'])
                    output_path = f"{OUTPUT_FOLDER}/card_{entry['ID']}.svg"
                    
                    generate_id(
                        TEMPLATE_FILE,
                        output_path,
                        entry['Name'], entry['Gmail'],
                        entry['ID'], entry['Blood Group '],
                        image_path
                    )
                    svg_to_pdf(output_path, f"{OUTPUT_PDF}/card_{entry['ID']}.pdf")

                    if test_mode and idx >= test_size:
                        break
        except KeyboardInterrupt as e:
            print(f"Error: {e}")
            with open('remaining_entries.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=entry.keys())
                if file.tell() == 0:
                    writer.writeheader()  # Write header only if file is empty
                for entry in extracted_data[idx:]:
                    writer.writerow(entry)

            exit(0)