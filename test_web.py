from backend.web_search import search_web_images

images = search_web_images(
    "Hyundai Venue red SUV"
)

print()

print("Found", len(images), "images")

print()

for img in images:

    print(img)