from typing import List, Dict
import aiohttp
from datetime import datetime
import yfinance as yf
from newspaper import Article, Config
from src.core.config import Settings
from src.services.news_publisher import NewsPublisher
import logging

logger = logging.getLogger(__name__)

class NewsService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.publisher = NewsPublisher()
        
    async def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news articles for a given stock symbol"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            news = yf.Ticker(symbol).news
            today_news = []
            
            for item in news:
                news_date = None
                if 'displayTime' in item:
                    news_date = item['displayTime'].split('T')[0]
                elif 'content' in item and 'displayTime' in item['content']:
                    news_date = item['content']['displayTime'].split('T')[0]
                
                if news_date == today:
                    article = {}
                    if 'content' in item:
                        content = item['content']
                        article['title'] = content.get('title', 'No Title')
                        article['date'] = news_date
                        article['summary'] = content.get('summary', '')
                        article['displayTime'] = content.get('displayTime', '')
                        if 'clickThroughUrl' in content and content['clickThroughUrl'] and 'url' in content['clickThroughUrl']:
                            article['link'] = content['clickThroughUrl']['url']
                        elif 'canonicalUrl' in content and content['canonicalUrl'] and 'url' in content['canonicalUrl']:
                            article['link'] = content['canonicalUrl']['url']
                        else:
                            article['link'] = ''
                        if 'provider' in content and content['provider']:
                            article['source'] = content['provider'].get('displayName', 'Unknown')
                        else:
                            article['source'] = 'Unknown'
                    else:
                        article['title'] = item.get('title', 'No Title')
                        article['date'] = news_date
                        article['summary'] = item.get('summary', '')
                        article['displayTime'] = item.get('displayTime', '')
                        article['link'] = item.get('link', '')
                        article['source'] = 'Unknown'
                    today_news.append(article)
            
            return today_news
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
            
    async def process_news(self, articles: List[Dict]) -> List[Dict]:
        """Process and clean news articles"""
        processed_articles = []
        for article in articles:
            try:
                if not article.get('link'):
                    continue
                    
                config = Config()
                config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
                config.request_timeout = 30
                
                news_article = Article(article['link'], config=config)
                news_article.download()
                news_article.parse()
                
                processed_article = {
                    "url": article['link'],
                    "title": news_article.title or article['title'],
                    "ticker": article.get('ticker', ''),
                    "content": news_article.text,
                    "date": article['date'],
                    "source": article.get('source', 'Unknown')
                }
                processed_articles.append(processed_article)
            except Exception as e:
                logger.error(f"Error processing article {article.get('link')}: {e}")
                continue
                
        return processed_articles

    async def get_news(self, symbol: str) -> List[Dict]:
        """Fetch, process and publish news articles"""
        articles = await self.fetch_news(symbol)
        processed_articles = await self.process_news(articles)
        
        # Publish articles to RabbitMQ
        for article in processed_articles:
            try:
                await self.publisher.publish_news(
                    title=article['title'],
                    symbol=symbol,
                    content=article['content'],
                    published_at=datetime.strptime(article['date'], '%Y-%m-%d'),
                    opinion=0  # Default to neutral, sentiment will be added later
                )
            except Exception as e:
                logger.error(f"Error publishing article {article['title']}: {e}")
                
        return processed_articles 