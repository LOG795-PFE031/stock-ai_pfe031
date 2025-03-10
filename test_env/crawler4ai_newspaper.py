import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult
import os

async def main():
    # Configure the browser (visible for debugging)
    browser_config = BrowserConfig(headless=False, verbose=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    # Target URL
    url = "https://finance.yahoo.com/news/palantir-stock-crash-2025-181700749.html"
    
    # Run the crawler
    result: CrawlResult = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            session_id="my_session_id",
            css_selector=".body.yf-tsvcyu",  # Selector for article content
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
            wait_for=".body.yf-tsvcyu",
            page_timeout=10000  # Increased to 180 seconds
        )
    )
    
    # Check and save the result
    if result and result.markdown:
        print(f"Extracted content length: {len(result.markdown.raw_markdown)}")
        with open("output.md", "w") as f:
            f.write(result.markdown.raw_markdown)
    else:
        print("Crawling failed, result is None or has no markdown")
    
    await crawler.close()

if __name__ == "__main__":
    asyncio.run(main())