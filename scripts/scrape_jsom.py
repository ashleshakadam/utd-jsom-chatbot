import asyncio
import aiohttp
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scripts.utils.helpers import (
    parse_sitemap_urls,
    clean_html,
    hash_url_content,
    ensure_directories_exist,
    log_error,
    load_json,
    save_json,
)

BASE_URL = "https://jindal.utdallas.edu"
SITEMAP_URL = f"{BASE_URL}/sitemap/"
RAW_DATA_PATH = "data/raw"
FULL_SITE_JSON = "data/full_site.json"
HASHES_PATH = "data/page_hashes.json"


async def fetch(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"Status {response.status}")
            return url, await response.text()
    except Exception as e:
        log_error(f"{url} — {str(e)}")
        return url, None


async def crawl_all(urls):
    crawled = {}
    connector = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, url) for url in urls]
        for future in asyncio.as_completed(tasks):
            url, html = await future
            if html:
                text = clean_html(html)
                crawled[url] = text
    return crawled


def main():
    print("🔍 Using sitemap to get URLs...")
    urls = parse_sitemap_urls(SITEMAP_URL)
    print(f"🔗 Found {len(urls)} URLs in sitemap.")

    ensure_directories_exist()

    # Load old hash map
    previous_hashes = load_json(HASHES_PATH)
    previous_data = load_json(FULL_SITE_JSON)

    print("⚡ Crawling site asynchronously...")
    crawled_data = asyncio.run(crawl_all(urls))

    latest_data = {}
    latest_hashes = {}

    for url, content in crawled_data.items():
        content_hash = hash_url_content(url, content)
        latest_hashes[url] = content_hash

        # Check if new or modified
        if url not in previous_hashes or content_hash != previous_hashes[url]:
            print(f"🔄 Updated: {url}")
            latest_data[url] = content
        else:
            latest_data[url] = previous_data.get(url, content)

    save_json(latest_data, FULL_SITE_JSON)
    save_json(latest_hashes, HASHES_PATH)

    print(f"✅ Crawl complete. Content saved to {FULL_SITE_JSON}")


if __name__ == "__main__":
    main()
