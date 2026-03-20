import feedparser
import requests
import time
from urllib.parse import quote
def fetch_company_news(company,limit=10,retries=3):
    query=quote(company)
    url=f"https://news.google.com/rss/search?q={query}"
    headers={
        "User-Agent":"Mozilla/5.0"
    }
    last_error=None
    for attempt in range(retries):
        try:
            response=requests.get(url,headers=headers,timeout=10)
            response.raise_for_status()
            feed=feedparser.parse(response.text)
            if getattr(feed,"bozo",False):
                print("Feed parse warning:",getattr(feed,"bozo_exception",None))
            headlines=[]
            for entry in feed.entries:
                title=getattr(entry,"title","")
                if isinstance(title,str) and title.strip():
                    headlines.append(title.strip())
                if len(headlines)>=limit:
                    break
            if headlines:
                return headlines
        except Exception as e:
            last_error=e
        time.sleep(1)
    print(f"Fetched {len(headlines)} headlines for {company}")
    return headlines