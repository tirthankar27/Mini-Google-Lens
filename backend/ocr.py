import easyocr

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("EasyOCR Ready")


def extract_text(image_path):

    results = reader.readtext(image_path)

    extracted = []

    for item in results:

        extracted.append(item[1])

    return extracted