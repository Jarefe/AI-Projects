import os
import cv2
import matplotlib.pyplot as plt
import pytesseract
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
# .env file should contain: TESSERACT_PATH=C:/Path/To/tesseract.exe
load_dotenv()

# Set the path to tesseract executable from .env file
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

# ------------------------------
# Function: Detect License Plate Number
# ------------------------------

def detect_plate_number(image_path):
    """
    Detect and extract the license plate number from an image using OpenCV and Tesseract OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Detected license plate number (if any), else a message.
    """

    # Step 1: Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")

    # Display the original image using matplotlib (in RGB format)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Step 2: Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     # Reduce noise
    edges = cv2.Canny(blurred, 100, 200)            # Edge detection

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate_contour = None

    # Step 4: Look for a rectangular contour (possible license plate)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 4 corners, it may be a license plate
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        # Step 5: Crop the license plate from the image
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]

        # Optional: Thresholding to improve OCR accuracy
        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 6: Use Tesseract to extract text
        # '--psm 8' tells Tesseract to treat the image as a single word
        plate_number = pytesseract.image_to_string(thresh, config='--psm 8')

        print("\nDetected License Plate Region:")
        plt.imshow(thresh, cmap='gray')
        plt.title("Detected Plate (Preprocessed for OCR)")
        plt.axis('off')
        plt.show()

        return plate_number.strip()

    else:
        return "No license plate contour detected."

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    image_to_process = 'car.png'
    result = detect_plate_number(image_to_process)
    print(f"Detected Plate Number: {result}")
