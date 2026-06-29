from urllib.parse import quote_plus


def shopping_links(ai_info):

    query = ai_info["search_query"]

    amazon = (
        "https://www.amazon.in/s?k="
        + quote_plus(query)
    )

    flipkart = (
        "https://www.flipkart.com/search?q="
        + quote_plus(query)
    )

    google = (
        "https://www.google.com/search?q="
        + quote_plus(query)
    )

    return {
        "amazon": amazon,
        "flipkart": flipkart,
        "google": google
    }