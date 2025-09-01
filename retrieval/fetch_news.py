import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GNEWS_API_KEY")
BASE_URL = "https://gnews.io/api/v4/search"

def fetch_gnews(query, max_results=10):
    params = {
        "q": query,
        "lang": "en",
        "country": "in",
        "max": max_results,
        "token": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    return response.json()