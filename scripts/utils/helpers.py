import os
import hashlib
import json
import re
from bs4 import BeautifulSoup
import requests

def parse_sitemap_urls(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.content, "xml")
        urls = [loc.text.strip() for loc in soup.find_all("loc")]
        return [url for url in urls if "https://jindal.utdallas.edu" in url]
    except Exception as e:
        print(f"❌ Failed to parse sitemap: {e}")
        return []

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def hash_url_content(url, content):
    combined = url.strip() + content.strip()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def ensure_directories_exist():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def log_error(message):
    with open("logs/crawl_errors.log", "a") as f:
        f.write(message + "\n")

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_json(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)
