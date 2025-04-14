from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Scraping de l'actualité
url = "https://news.ycombinator.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
articles = [a.text for a in soup.select(".titleline > a")[:5]]

# Analyse de sentiment
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
for article in articles:
    result = sentiment_analyzer(article)[0]
    print(f"Article: {article}\nSentiment: {result['label']} (Score: {result['score']:.2f})\n")