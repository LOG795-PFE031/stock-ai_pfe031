import React, { useEffect, useState } from 'react';
import {
  Box,
  Heading,
  Text,
  Flex,
  Button,
  Spinner,
  Alert,
  AlertIcon,
  AlertDescription,
  Badge,
  Progress,
  Container,
  SimpleGrid
} from "@chakra-ui/react";
import NewsCard from '../models/NewsCard'
import apiService, { Stock } from '../clients/ApiService';

type NewsData = {
  title: string;
  publishedAt: string;
  opinion: number;
  url?: string;
  source?: string;
  confidence?: number;
};

type params = {
  searchTerm: string;
  isLoggedIn: boolean
};

// Cache interface
interface NewsCache {
  data: NewsData[];
  timestamp: number;
  metrics?: {
    positive: number;
    negative: number;
    neutral: number;
  };
}

// Cache storage
const newsCache: Record<string, NewsCache> = {};

const News: React.FC<params> = ({ searchTerm, isLoggedIn }) => {
  const [newsData, setNewsData] = useState<NewsData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTicker, setCurrentTicker] = useState<string>(searchTerm);
  const [searchInput, setSearchInput] = useState<string>(searchTerm);
  const [sentimentMetrics, setSentimentMetrics] = useState<{ positive: number, negative: number, neutral: number } | null>(null);
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);

  useEffect(() => {
    const loadStocks = async () => {
      const allStocks = localStorage.getItem('nasdaqStocks');
      if (allStocks) {
        const stocksList: Stock[] = JSON.parse(allStocks)
        stocksList.sort((a, b) => {
          if (a.symbol < b.symbol) return -1;
          else if (a.symbol > b.symbol) return 1;
          return 0
        })
        setStocks(stocksList);
        setCurrentTicker(stocksList[0].symbol)
        setSearchInput(stocksList[0].symbol)
      }
      else {
        try {
          const stocksData = await apiService.getStocks();
          stocksData.sort((a: Stock, b: Stock) => {
            if (a.symbol < b.symbol) return -1;
            else if (a.symbol > b.symbol) return 1;
            return 0
          })
          setStocks(stocksData)
          setCurrentTicker(stocksData[0].symbol)
          setSearchInput(stocksData[0].symbol)
          localStorage.setItem('nasdaqStocks', JSON.stringify(stocksData));
        } catch (error) {
          console.error("Failed to load stocks:", error);
          setError("Failed to load stocks. Please try again later.");
        }
      }
    }
    loadStocks()
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      console.log(`Attempting to fetch news for ${currentTicker}`);
      try {

        // Check if we have cached data for this stock

        const cachedData = newsCache[currentTicker];
        const now = Date.now();

        // If we have cached data from today, use it
        if (cachedData && now - cachedData.timestamp < 24 * 60 * 60 * 1000) {
          console.log(`Using cached data for ${currentTicker}`);
          setNewsData(cachedData.data);
          if (cachedData.metrics) {
            setSentimentMetrics(cachedData.metrics);
          }
          setLoading(false);
          return;
        }

        console.log(`Calling API for ${currentTicker} news data`);
        // Otherwise fetch new data from the API service
        const response = await apiService.getNewsData(currentTicker);
        console.log(`API Response for ${currentTicker}:`, response);

        // Transform the data to match our NewsData type
        if (!response.articles || !Array.isArray(response.articles)) {
          console.error(`Invalid response format for ${currentTicker}:`, response);
          throw new Error('Invalid response format: missing articles array');
        }

        const newData = response.articles.map(article => ({
          title: article.title || 'No title available',
          publishedAt: article.publishedAt || new Date().toISOString(),
          opinion: typeof article.opinion === 'number' ? article.opinion : 0,
          url: article.url || '',
          source: article.source || '',
          confidence: article.confidence || 0
        }));

        console.log(`Transformed data for ${currentTicker}:`, newData);

        // Get sentiment metrics if available
        const metrics = response.sentiment_metrics ? {
          positive: response.sentiment_metrics.positive || 0,
          negative: response.sentiment_metrics.negative || 0,
          neutral: response.sentiment_metrics.neutral || 0
        } : null;

        if (metrics) {
          setSentimentMetrics(metrics);
        }

        // Update cache
        newsCache[currentTicker] = {
          data: newData,
          timestamp: now,
          metrics: metrics || undefined
        };

        setNewsData(newData);
        setError(null);
      } catch (error) {
        console.error(`Error fetching news for ${currentTicker}:`, error);
        if (error instanceof Error) {
          setError(`Failed to fetch news data: ${error.message}`);
        } else {
          setError(`Failed to fetch news data for ${currentTicker}. Please try again later.`);
        }
        setNewsData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [currentTicker]);

  const handleSearchInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setSearchInput(value);
    
    // Filter stocks based on search input
    if (value.trim() !== '') {
      const filtered = stocks.filter(stock => 
        stock.symbol.includes(value) || 
        stock.companyName.toUpperCase().includes(value)
      );
      setFilteredStocks(filtered.slice(0, 5)); // Limit to 5 suggestions
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleSelectStock = (symbol: string) => {
    setSearchInput(symbol);
    setCurrentTicker(symbol);
    setShowSuggestions(false);
    updateStockSelected(symbol);
  };

  const updateStockSelected = async (ticker: string) => {
    setLoading(true)
    try {
      // Validate if ticker exists in stocks array
      const isValidTicker = stocks.some(stock => stock.symbol === ticker);
      if (!isValidTicker) {
        throw new Error(`Invalid stock symbol: ${ticker}`);
      }

      // Fetch news data from the API service
      const response = await apiService.getNewsData(ticker);
      console.log(`API Response for ${ticker}:`, response);

      // Transform the data to match our NewsData type
      if (!response.articles || !Array.isArray(response.articles)) {
        console.error(`Invalid response format for ${ticker}:`, response);
        throw new Error('Invalid response format: missing articles array');
      }

      const newData = response.articles.map(article => ({
        title: article.title || 'No title available',
        publishedAt: article.publishedAt || new Date().toISOString(),
        opinion: typeof article.opinion === 'number' ? article.opinion : 0,
        url: article.url || '',
        source: article.source || '',
        confidence: article.confidence || 0
      }));

      setNewsData(newData);
      setCurrentTicker(ticker);
      setSearchInput(ticker);

      console.log(`Transformed data for ${ticker}:`, newData);

      // Get sentiment metrics if available
      const metrics = response.sentiment_metrics ? {
        positive: response.sentiment_metrics.positive || 0,
        negative: response.sentiment_metrics.negative || 0,
        neutral: response.sentiment_metrics.neutral || 0
      } : null;

      if (metrics) {
        setSentimentMetrics(metrics);
      }

      const now = Date.now();
      // Update cache
      newsCache[ticker] = {
        data: newData,
        timestamp: now,
        metrics: metrics || undefined
      };

      
      
      setLoading(false)
    }
    catch (error) {
      console.error(`Error fetching news for ${ticker}:`, error);
      if (error instanceof Error) {
        setError(`Failed to fetch news data: ${error.message}`);
      } else {
        setError(`Failed to fetch news data for ${ticker}. Please try again later.`);
      }
      setNewsData([]);
      setLoading(false)
    }


  }

  if (loading) return (
    <Flex justify="center" align="center" height="300px">
      <Spinner size="xl" color="brand.500" thickness="4px" mr={4} />
      <Text fontSize="xl" fontWeight="medium" color="gray.700">Loading news...</Text>
    </Flex>
  );

  if (error) return (
    <Alert status="error" variant="left-accent" borderRadius="md" mb={4}>
      <AlertIcon />
      <AlertDescription>{error}</AlertDescription>
    </Alert>
  );

  if (newsData.length === 0) return (
    <Alert status="info" variant="left-accent" borderRadius="md">
      <AlertIcon />
      <AlertDescription>No news articles found for {currentTicker}.</AlertDescription>
    </Alert>
  );

  return (
    <Container maxW="7xl" px={[2, 4, 6]} py={6}>
      <Box mb={6}>
        <Flex direction={["column", "row"]} justify="space-between" align={["flex-start", "center"]} mb={4}>
          <Heading as="h1" size="lg" mb={[3, 0]} color="gray.800">
            Latest News for {currentTicker}
          </Heading>

          {/* Ticker search */}
          <Flex wrap="wrap" gap={2} align="center">
            <Box position="relative" width="250px">
              <input
                type="text"
                placeholder="Search stock symbol..."
                value={searchInput}
                onChange={handleSearchInputChange}
                style={{ 
                  width: '100%', 
                  padding: '8px 12px',
                  borderRadius: '15px',
                  border: '1px solid #e2e8f0',
                  fontSize: '14px'
                }}
              />
              <Button
                size="sm"
                colorScheme="blue"
                ml={2}
                onClick={() => searchInput && handleSelectStock(searchInput)}
                style={{
                  position: 'absolute',
                  right: '4px',
                  top: '4px',
                  zIndex: 2,
                  borderRadius: '12px',
                  padding: '0 12px',
                  height: '28px'
                }}
              >
                Search
              </Button>
              
              {/* Suggestions dropdown */}
              {showSuggestions && filteredStocks.length > 0 && (
                <Box
                  position="absolute"
                  top="40px"
                  left="0"
                  width="100%"
                  bg="white"
                  borderRadius="md"
                  boxShadow="md"
                  zIndex="10"
                  maxHeight="200px"
                  overflowY="auto"
                >
                  {filteredStocks.map((stock) => (
                    <Box
                      key={stock.symbol}
                      p={2}
                      _hover={{ bg: "gray.100" }}
                      cursor="pointer"
                      onClick={() => handleSelectStock(stock.symbol)}
                    >
                      <Text fontWeight="bold">{stock.symbol}</Text>
                      <Text fontSize="xs" color="gray.600">{stock.companyName}</Text>
                    </Box>
                  ))}
                </Box>
              )}
            </Box>
          </Flex>
        </Flex>

        {/* Account status indicator */}
        {!isLoggedIn && (
          <Alert status="warning" variant="left-accent" mb={4} py={2} fontSize="sm">
            <AlertIcon />
            <AlertDescription>
              Sign in to save your favorite tickers and get personalized news recommendations.
            </AlertDescription>
          </Alert>
        )}

        {/* Sentiment metrics */}
        {sentimentMetrics && (
          <Box bg="white" borderRadius="lg" boxShadow="sm" p={4} mb={4}>
            <Heading as="h3" size="sm" mb={3} pb={2} borderBottom="1px" borderColor="gray.200">
              Sentiment Analysis
            </Heading>
            <SimpleGrid columns={[1, 3]} spacing={3}>
              <Box bg="green.50" borderRadius="md" p={2}>
                <Flex justify="space-between" mb={1}>
                  <Text fontSize="xs" fontWeight="medium" color="gray.700">Positive</Text>
                  <Badge colorScheme="green" fontSize="xs" px={1.5}>
                    {Math.round(sentimentMetrics.positive * 100)}%
                  </Badge>
                </Flex>
                <Progress
                  value={Math.max(5, sentimentMetrics.positive * 100)}
                  size="xs"
                  colorScheme="green"
                  borderRadius="full"
                />
              </Box>
              <Box bg="gray.50" borderRadius="md" p={2}>
                <Flex justify="space-between" mb={1}>
                  <Text fontSize="xs" fontWeight="medium" color="gray.700">Neutral</Text>
                  <Badge colorScheme="gray" fontSize="xs" px={1.5}>
                    {Math.round(sentimentMetrics.neutral * 100)}%
                  </Badge>
                </Flex>
                <Progress
                  value={Math.max(5, sentimentMetrics.neutral * 100)}
                  size="xs"
                  colorScheme="gray"
                  borderRadius="full"
                />
              </Box>
              <Box bg="red.50" borderRadius="md" p={2}>
                <Flex justify="space-between" mb={1}>
                  <Text fontSize="xs" fontWeight="medium" color="gray.700">Negative</Text>
                  <Badge colorScheme="red" fontSize="xs" px={1.5}>
                    {Math.round(sentimentMetrics.negative * 100)}%
                  </Badge>
                </Flex>
                <Progress
                  value={Math.max(5, sentimentMetrics.negative * 100)}
                  size="xs"
                  colorScheme="red"
                  borderRadius="full"
                />
              </Box>
            </SimpleGrid>
          </Box>
        )}

        <Box bg="white" borderRadius="lg" boxShadow="sm" p={3} mb={4}>
          <Flex justify="space-between" align="center">
            <Text fontSize="sm" color="gray.600">
              <Text as="span" fontWeight="medium" color="gray.800">{newsData.length}</Text> articles found
            </Text>
            <Text fontSize="xs" color="gray.500">
              Last updated: {new Date().toLocaleString()}
            </Text>
          </Flex>
        </Box>
      </Box>

      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
        {newsData.map((news, index) => (
          <Box key={index} transform="auto" _hover={{ translateY: "-1px" }} transition="transform 0.3s">
            <NewsCard data={news} />
          </Box>
        ))}
      </SimpleGrid>
    </Container>
  );
}

export default News;
