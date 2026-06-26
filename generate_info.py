import json
import os

# Load class names
with open("classes.json", "r") as f:
    classes = json.load(f)

object_info = {}

for cls in classes:

    # Remove numeric prefix
    clean_name = cls.split(".", 1)[1]

    clean_name = clean_name.replace("-", " ")

    clean_name = clean_name.title()

    object_info[clean_name] = {
        "category": "Unknown",
        "description": "Information not available."
    }

os.makedirs("data", exist_ok=True)

with open("data/object_info.json", "w") as f:
    json.dump(object_info, f, indent=4)

print("Created object_info.json")