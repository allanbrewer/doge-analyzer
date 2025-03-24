import requests
from bs4 import BeautifulSoup
import re
import json

url = "https://doge.gov/savings"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "text/html",
    "Cookie": "__cf_bm=LSAROZQkfOm3Zhl0_jiAX6b_pguL8uUGW2U03wEtZi0-1742695473-1.0.1.1-...",
}
response = requests.get(url, headers=headers)
html_content = response.text

# Find all script tags
soup = BeautifulSoup(html_content, "html.parser")
script_tags = soup.find_all("script")

# Extract savings data
contracts = []
for script in script_tags:
    if script.string and "__next_f.push" in script.string:
        script_content = script.string
        # Extract all push payloads
        payloads = re.findall(
            r"self\.__next_f\.push\(\[(.*?)\]\)", script_content, re.DOTALL
        )
        for payload in payloads:
            # Look for contract-like data
            if any(
                key in payload for key in ["description", "value", "savings", "href"]
            ):
                # Clean and attempt JSON parsing
                try:
                    # Remove Next.js markers and fix syntax
                    cleaned_payload = (
                        payload.split(":", 1)[1] if ":" in payload else payload
                    )
                    cleaned_payload = (
                        cleaned_payload.replace('"$', '"')
                        .replace("$L", "")
                        .replace("null", '"null"')
                    )
                    # Wrap in brackets if it’s a single object or array
                    if not cleaned_payload.strip().startswith("["):
                        cleaned_payload = f"[{cleaned_payload}]"
                    data = json.loads(cleaned_payload)

                    # Extract contract objects
                    def extract_contracts(obj):
                        if isinstance(obj, dict):
                            if "description" in obj and "value" in obj:
                                contracts.append(
                                    {
                                        "description": obj.get("description"),
                                        "link": obj.get("link") or obj.get("href"),
                                        "value": obj.get("value"),
                                        "savings": obj.get("savings"),
                                    }
                                )
                            for value in obj.values():
                                extract_contracts(value)
                        elif isinstance(obj, list):
                            for item in obj:
                                extract_contracts(item)

                    extract_contracts(data)
                except json.JSONDecodeError as e:
                    print(
                        f"Parsing error in payload: {e}. Raw content: {payload[:200]}"
                    )

# Save results
if contracts:
    with open("savings_contracts.json", "w") as f:
        json.dump(contracts, f, indent=2)
    print(f"Extracted {len(contracts)} contracts")
else:
    print(
        "No contracts found. Check additional script tags or share a sample with contract data."
    )
