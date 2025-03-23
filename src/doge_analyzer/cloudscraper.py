import cloudscraper

urls = [
    "https://doge.gov/api/ping",
    "https://doge.gov/api/health",
    "https://doge.gov/api/v1/ping",
    "https://doge.gov/api/savings",
]
scraper = cloudscraper.create_scraper()
for url in urls:
    response = scraper.get(url)
    print(f"{url}: {response.status_code}")
    if response.status_code == 200:
        print(response.text)
