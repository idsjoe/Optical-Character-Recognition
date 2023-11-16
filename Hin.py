import pytesseract
from PIL import Image
img_file = "Pictures/Hindi.jpg"
img = Image.open(img_file)
ocr_result = pytesseract.image_to_string(img, lang="hin")
print(ocr_result)