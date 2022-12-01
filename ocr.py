try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text


print(ocr_core('5397.jpg'))
