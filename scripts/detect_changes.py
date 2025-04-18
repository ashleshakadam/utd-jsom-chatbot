# File: scripts/detect_changes.py
import os
import json
from scripts.utils.hashing import hash_content
from scripts.utils.io import load_json, save_json

DATA_PATH = "data"
CURRENT_SNAPSHOT = os.path.join(DATA_PATH, "full_site.json")
PREVIOUS_HASHES = os.path.join(DATA_PATH, "page_hashes.json")
CHANGES_FILE = os.path.join(DATA_PATH, "changes.json")

# Load current data and old hashes
current_data = load_json(CURRENT_SNAPSHOT)
old_hashes = load_json(PREVIOUS_HASHES)

# Detect changes
new_hashes = {}
added, updated, removed = [], [], []

for url, content in current_data.items():
    new_hash = hash_content(url, content)
    new_hashes[url] = new_hash

    if url not in old_hashes:
        added.append(url)
    elif old_hashes[url] != new_hash:
        updated.append(url)

for url in old_hashes:
    if url not in new_hashes:
        removed.append(url)

# Save diffs
diffs = {
    "added": added,
    "updated": updated,
    "removed": removed
}
save_json(diffs, CHANGES_FILE)
save_json(new_hashes, PREVIOUS_HASHES)
print("✅ Change detection completed.")
