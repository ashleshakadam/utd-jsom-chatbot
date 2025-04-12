import requests
from bs4 import BeautifulSoup
import os

# URL to scrape
URL = "https://jindal.utdallas.edu/programs/masters-programs/"

# Fetch the page
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# Clean unwanted tags
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()

# Extract main visible text
text = soup.get_text(separator="\n", strip=True)

# Save to data/admissions.txt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(project_root, "data", "admissions.txt")

# Only save if content exists
if text.strip():
    with open(output_path, "w") as f:
        f.write(text)
    print("✅ Content saved to admissions.txt")
else:
    print("❌ No content found. Check if the page loaded correctly.")
