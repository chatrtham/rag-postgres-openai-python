import os

import requests
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

def google_search_function(search_query):
    # Replace with your actual API key
    api_key = os.environ["GOOGLE_SEARCH_API_KEY"]

    # Custom search engine ID
    cx = os.environ["GOOGLE_SEARCH_ENGINE_ID"]

    # Construct the URL
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={search_query}"

    # Send the GET request
    response = requests.get(url)

    links = []
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        for item in data.get("items", []):
            link = item.get("link")
            if link:
                links.append(link)
        return links
    else:
        return {"error": "An error occurred"}


if __name__ == "__main__":
    search_query = "ลดขนาดหน้าอก "
    result = google_search_function(search_query)
    print(result)
