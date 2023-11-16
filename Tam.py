import pytesseract
from PIL import Image
img_file = "Pictures/3-Figure2-1.png"
img = Image.open(img_file)
ocr_result = pytesseract.image_to_string(img, lang="tam")
print(ocr_result)