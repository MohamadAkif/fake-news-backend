import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    """
    Scrape the main text content from the given URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all paragraphs and combine their text
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.text for p in paragraphs if p.text])
        return article_text.strip()
    except Exception as e:
        raise RuntimeError(f"Error scraping URL: {str(e)}")
