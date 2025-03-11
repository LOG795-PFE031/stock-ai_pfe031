import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult
from bs4 import BeautifulSoup
import json
from typing import List, Dict
from deepcrawler import get_todays_news_urls
# Define the JavaScript code to handle consent banner and wait for article body
JS_CODE = """
(async () => {
    try {
        // Wait for consent banner, max 10 seconds
        await new Promise((resolve) => {
            const startTime = Date.now();
            const checkBanner = () => {
                const banner = document.querySelector("#consent-page, .consent-form, .consent-container");
                if (banner) resolve();
                else if (Date.now() - startTime > 10000) resolve(); // Timeout after 10s
                else setTimeout(checkBanner, 100);
            };
            checkBanner();
        });

        // Find and click the accept button
        const acceptButton = document.querySelector(
            'button[name="agree"], button.accept-all, button.btn.btn_secondary_accept-all'
        );
        if (acceptButton) {
            acceptButton.click();
            console.log("Accept button clicked");
            // Wait 5 seconds for the page to load after clicking
            await new Promise(resolve => setTimeout(resolve, 5000));
        } else {
            console.log("Accept button not found");
        }

        // Wait for the article body with dynamic class, max 60 seconds
        await new Promise((resolve) => {
            const startTime = Date.now();
            const checkArticle = () => {
                const articleBody = document.querySelector("[class^='body yf-']");
                if (articleBody) {
                    console.log("Article body found");
                    resolve();
                } else if (Date.now() - startTime > 60000) {
                    console.log("Article body not found within 60 seconds");
                    resolve();
                } else {
                    setTimeout(checkArticle, 100);
                }
            };
            checkArticle();
        });
    } catch (error) {
        console.error("Error in js_code: " + error.message);
    }
})();
"""

async def crawl_and_extract(crawler: AsyncWebCrawler, url: str, ticker: str) -> Dict:
    """
    Crawl a single URL and extract title, date, and content.
    
    Args:
        crawler (AsyncWebCrawler): The crawler instance.
        url (str): The URL to crawl.
    
    Returns:
        Dict: A dictionary containing the URL, title, date, content, or error message.
    """
    try:
        result: CrawlResult = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                session_id="yahoo_crawler_session",
                js_code=[JS_CODE],
                wait_for="[class^='body yf-']",
                # css_selector is set to None to get the full page HTML
            )
        )

        if not result or not hasattr(result, 'html') or not result.html:
            return {"url": url, "error": "Failed to retrieve HTML"}

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(result.html, 'html.parser')

        # Extract title (using 'h1.cover-headline', assuming it's consistent)
        title_elem = soup.select_one('.cover-headline.yf-1rjrr1')
        title = title_elem.get_text().strip() if title_elem else None
        if not title_elem:
            print(f"Title not found for {url}")

        # Extract date (using 'span.byline-attr-meta-time', assuming it's consistent)
        date_elem = soup.select_one('.byline-attr-meta-time')
        date = date_elem.get_text().strip() if date_elem else None
        if not date_elem:
            print(f"Date not found for {url}")

        # Extract content (using '[class^="body yf-"]' for dynamic class)
        content_elem = soup.select_one('[class^="body yf-"]')
        article_text = (
            '\n\n'.join([p.get_text().strip() for p in content_elem.find_all('p')])
            if content_elem
            else None
        )
        if not content_elem:
            print(f"Content not found for {url}")

        return {
            "url": url,
            "title": title,
            "ticker": ticker,
            "date": date,
            "content": article_text
        }

    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return {"url": url, "error": str(e)}

async def main(urls: List[str], ticker: str):
    """
    Main function to crawl multiple URLs and save results to a JSON file.
    
    Args:
        urls (List[str]): List of URLs received from deepcrawler.py.
    """
    # Filter URLs to include only those from Yahoo Finance
    yahoo_urls = [url for url in urls if "finance.yahoo.com" in url]
    print(f"Found {len(yahoo_urls)} Yahoo Finance URLs to crawl.")

    # Configure the browser
    browser_config = BrowserConfig(headless=False, verbose=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    # Crawl all URLs concurrently
    tasks = [crawl_and_extract(crawler, url, ticker) for url in yahoo_urls]
    results = await asyncio.gather(*tasks)

    # Save results to a JSON file
    with open("articles.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Saved {len(results)} articles to 'articles.json'.")

    # Clean up
    await crawler.close()

if __name__ == "__main__":
    # Example list of URLs (in practice, this would come from deepcrawler.py)

    #read the ticker from the user on CLI
    ticker = input("Enter the ticker: ")

    urls, ticker = get_todays_news_urls(ticker)
    asyncio.run(main(urls, ticker))