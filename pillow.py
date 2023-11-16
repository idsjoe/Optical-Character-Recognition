from PIL import Image
im_file = "Images/image2.jpg"

im = Image.open(im_file)
im.rotate(15).show()
im.save("Pictures/image2.jpg")