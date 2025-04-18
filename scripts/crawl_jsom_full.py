import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

# Base config
BASE_DOMAIN = "jindal.utdallas.edu"
BASE_URL = "https://jindal.utdallas.edu"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "full_site.txt")

# Crawl state
visited = set()
content_chunks = []

def is_valid_url(href):
    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return False
    parsed = urlparse(href)
    if parsed.netloc and BASE_DOMAIN not in parsed.netloc:
        return False
    if parsed.path.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.mp4', '.zip', '.docx')):
        return False
    return True

def clean_soup(soup):
    for tag in soup(["script", "style", "footer", "nav", "form", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def crawl(url):
    global visited

    if url in visited:
        return
    visited.add(url)

    try:
        print(f"🔍 Visiting: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract and store content
        text = clean_soup(soup)
        if text.strip():
            content_chunks.append(f"\n\n--- {url} ---\n\n{text}")

        # Follow internal links
        for link_tag in soup.find_all("a", href=True):
            href = link_tag['href']
            if not is_valid_url(href):
                continue
            full_url = urljoin(url, href.split("#")[0])
            crawl(full_url)

        time.sleep(1)  # Polite crawling
    except Exception as e:
        print(f"❌ Failed to crawl {url} — {e}")

def save_results():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(content_chunks))
    print(f"\n✅ Crawled {len(visited)} pages. Content saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl(BASE_URL)
    save_results()
