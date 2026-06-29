from ddgs import DDGS
import requests
from PIL import Image
from io import BytesIO

cache = {}

def valid_image(url):

    try:

        r = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            return False

        img = Image.open(BytesIO(r.content))

        w, h = img.size

        if w < 300 or h < 300:
            return False

        return True

    except:
        return False


def search_web_images(query, limit=12):

    with DDGS() as ddgs:

        results = ddgs.images(
            query=query,
            max_results=limit
        )

        images = []

        for item in results:

            if "image" not in item:
                continue

            url = item["image"]

            if valid_image(url):

                images.append(url)

            if len(images) == 6:
                break
        cache[query] = images
        return images