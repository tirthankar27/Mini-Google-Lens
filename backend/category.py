def detect_category(text):

    text = text.lower()

    PHONE = [
        "iphone",
        "samsung",
        "pixel",
        "oneplus",
        "xiaomi",
        "mobile",
        "smartphone"
    ]

    VEHICLE = [
        "car",
        "truck",
        "bus",
        "bike",
        "motorcycle",
        "jeep",
        "suv",
        "sedan"
    ]

    LAPTOP = [
        "laptop",
        "macbook",
        "notebook"
    ]

    WATCH = [
        "watch",
        "smartwatch"
    ]

    CAMERA = [
        "camera",
        "dslr"
    ]

    SHOE = [
        "shoe",
        "sneaker",
        "boot"
    ]

    for word in PHONE:
        if word in text:
            return "phone"

    for word in VEHICLE:
        if word in text:
            return "vehicle"

    for word in LAPTOP:
        if word in text:
            return "laptop"

    for word in WATCH:
        if word in text:
            return "watch"

    for word in CAMERA:
        if word in text:
            return "camera"

    for word in SHOE:
        if word in text:
            return "shoe"

    return "unknown"