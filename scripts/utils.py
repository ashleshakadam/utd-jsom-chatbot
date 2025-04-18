# File: scripts/scrape_jsom.py

import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from scripts.utils import (
    hash_url_content,
    ensure_directories_exist,
    log_error,
    save_json,
    load_json,
)

BASE_URL = "https://jindal.utdallas.edu"
CRAWLED_URLS = set()
MAX_DEPTH = 3

# Load previously crawled content hash
previous_hashes = load_json("data/hashes_previous.json")
latest_hashes = {}

# Ensure necessary directories
ensure_directories_exist()

# Store new content for diffing
latest_data = {}


def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and BASE_URL in url


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    links = set()
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a_tag['href'])
        if is_valid_url(full_url):
            links.add(full_url.split("#")[0])  # Remove fragments
    return links, soup.get_text(separator="\n", strip=True)


def crawl(url, depth):
    if depth > MAX_DEPTH or url in CRAWLED_URLS:
        return
    try:
        print(f"🔍 Visiting: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        links, visible_text = extract_links(resp.text, url)

        url_hash = hash_url_content(url, visible_text)
        latest_hashes[url] = url_hash
        latest_data[url] = visible_text

        CRAWLED_URLS.add(url)

        for link in links:
            crawl(link, depth + 1)

    except Exception as e:
        print(f"❌ Failed to crawl {url} — {e}")
        log_error(f"{url} — {e}")


if __name__ == "__main__":
    print("\n🔍 Crawling site and capturing latest state...")
    crawl(BASE_URL, depth=0)

    print("\n💾 Saving current snapshot and hashes...")
    save_json(latest_data, "data/latest_data.json")
    save_json(latest_hashes, "data/hashes_latest.json")
    print("✅ Done!")
