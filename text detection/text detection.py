import cv2
import pytesseract
from matplotlib import pyplot as plt

# Read image using OpenCV (BGR format by default)
image = cv2.imread('sample.jpg')

# Check if image is successfully loaded
if image is None:
    raise FileNotFoundError("The image 'sample.jpg' was not found. Make sure the file path is correct.")

# Convert image to RGB format for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Grayscale image created.')

# Display grayscale image using OpenCV (press any key to close)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)            # Wait for a key press
cv2.destroyAllWindows()   # Close the OpenCV window

# Display original RGB image using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title('Original Image (RGB)')
plt.axis('off')
plt.tight_layout()
plt.show()

extracted_text = pytesseract.image_to_string(image_rgb)
print(f'Extracted Text:\n{extracted_text}')

# Below code does not function without tesseract installed
# # Draw bounding boxes around detected text
#
# data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
#
# n_boxes = len(data['level'])
# for i in range(n_boxes):
#     (x,y,w,h) = (data['left'][i], data['top'][i], data['width'][i],data['height'][i])
#     cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (255,0,0), 2)
#
#
# # Show image with bounding boxes
#
# plt.figure(figsize=(10, 6))
# plt.imshow(image_rgb)
# plt.title("Image with Text Bounding Boxes")
# plt.axis("off")
# plt.show()