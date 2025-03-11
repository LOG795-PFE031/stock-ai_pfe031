import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult
import os
from bs4 import BeautifulSoup

async def main():
    # Configure the browser (visible for debugging)
    browser_config = BrowserConfig(headless=True, verbose=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    # Target URL
    url = "https://finance.yahoo.com/news/palantir-stock-crash-2025-181700749.html"
    
    # Run the crawler
    result: CrawlResult = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            session_id="my_session_id",
            css_selector=".article.yf-l7apfj",  # Selector for article content
            js_code=["""
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

                    // Wait for the article body, max 60 seconds
                    await new Promise((resolve) => {
                        const startTime = Date.now();
                        const checkArticle = () => {
                            const articleBody = document.querySelector(".caas-body");
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
            """],
            wait_for=".article.yf-l7apfj",
        )
    )
    
    # Check and save the result
    if result:
        print(f"Raw HTML length: {len(result.html) if hasattr(result, 'html') and result.html else 0}")
        
        # Process the HTML directly using BeautifulSoup to extract elements with the provided selectors
        # Create soup object from HTML content
        soup = BeautifulSoup(result.html, 'html.parser') if hasattr(result, 'html') and result.html else None
        
        # Extract article components using the provided CSS selectors
        title = None
        published_date = None
        article_text = None
        
        if soup:
            # Extract title using selector: "cover-headline yf-1rjrr1"
            title_elem = soup.select_one('.cover-headline.yf-1rjrr1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract published date using selector: "byline-attr-meta-time"
            date_elem = soup.select_one('.byline-attr-meta-time')
            if date_elem:
                published_date = date_elem.get_text().strip()
            
            # Extract article content using selector: "body yf-tsvcyu"
            content_elem = soup.select_one('.body.yf-tsvcyu')
            if content_elem:
                # Get all paragraphs within the content
                paragraphs = content_elem.find_all('p')
                article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
        
        # Print the extracted components
        print("\n" + "="*50)
        print("ARTICLE TITLE:", title if title else "Not found")
        print("="*50)
        print("DATE PUBLISHED:", published_date if published_date else "Not found")
        print("="*50)
        print("ARTICLE TEXT:")
        print(article_text if article_text else "Not found")
        print("="*50 + "\n")
        
        # Save the extracted content to a file
        with open("article_extraction.txt", "w") as f:
            f.write(f"TITLE: {title if title else 'Not found'}\n\n")
            f.write(f"DATE: {published_date if published_date else 'Not found'}\n\n")
            f.write(f"CONTENT:\n{article_text if article_text else 'Not found'}")
            
        # Still save the markdown if available
        if hasattr(result, 'markdown') and result.markdown:
            with open("output.md", "w") as f:
                f.write(result.markdown.raw_markdown)
    else:
        print("Crawling failed, result is None")
    
    await crawler.close()

if __name__ == "__main__":
    asyncio.run(main())

