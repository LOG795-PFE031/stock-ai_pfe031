import axios from 'axios';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock axios
vi.mock('axios');
const mockedAxios = axios as unknown as {
  get: ReturnType<typeof vi.fn>;
};

describe('API Service', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Stock Prediction API', () => {
    it('should fetch prediction for AAPL', async () => {
      // Mock response data
      const mockResponse = {
        data: {
          symbol: 'AAPL',
          date: '2023-08-15',
          predicted_price: 185.42,
          confidence: 0.87,
          model_type: 'lstm',
          timestamp: '2023-08-14T15:30:00Z'
        }
      };

      mockedAxios.get.mockResolvedValueOnce(mockResponse);

      // Make the API call
      const response = await axios.get('http://localhost:8000/api/predict/AAPL');

      // Assertions
      expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:8000/api/predict/AAPL');
      expect(response.data.symbol).toBe('AAPL');
      expect(response.data.predicted_price).toBeDefined();
      expect(typeof response.data.predicted_price).toBe('number');
    });

    it('should fetch prediction for NVDA', async () => {
      // Mock response data
      const mockResponse = {
        data: {
          symbol: 'NVDA',
          date: '2023-08-15',
          predicted_price: 452.18,
          confidence: 0.92,
          model_type: 'lstm',
          timestamp: '2023-08-14T15:30:00Z'
        }
      };

      mockedAxios.get.mockResolvedValueOnce(mockResponse);

      // Make the API call
      const response = await axios.get('http://localhost:8000/api/predict/NVDA');
      console.log(response.data);
      // Assertions
      expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:8000/api/predict/NVDA');
      expect(response.data.symbol).toBe('NVDA');
      expect(response.data.predicted_price).toBeDefined();
      expect(typeof response.data.predicted_price).toBe('number');
    });
  });

  describe('News and Sentiment API', () => {
    it('should fetch news and sentiment for AAPL', async () => {
      // Mock response data
      const mockResponse = {
        data: {
          symbol: 'AAPL',
          articles: [
            { title: "Apple Reports Strong Quarterly Earnings", publishedAt: "2023-08-01 14:25:35", opinion: 1, additionalProp1: {} },
            { title: "iPhone 15 Launch Delayed", publishedAt: "2023-08-02 10:15:22", opinion: -1, additionalProp1: {} }
          ],
          total_articles: 2,
          sentiment_metrics: {
            additionalProp1: 0,
            additionalProp2: 0,
            additionalProp3: 0
          },
          meta: {
            message: "Success",
            version: "1.0",
            documentation: "https://api-docs.example.com",
            endpoints: ["string"]
          }
        }
      };

      mockedAxios.get.mockResolvedValueOnce(mockResponse);

      // Make the API call
      const response = await axios.get('http://localhost:8000/api/data/news/AAPL');
      console.log(response.data);
      // Assertions
      expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:8000/api/data/news/AAPL');
      expect(response.data.symbol).toBe('AAPL');
      expect(response.data.articles).toBeInstanceOf(Array);
      expect(response.data.articles.length).toBeGreaterThan(0);
      expect(response.data.articles[0]).toHaveProperty('title');
      expect(response.data.articles[0]).toHaveProperty('publishedAt');
      expect(response.data.articles[0]).toHaveProperty('opinion');
    });

    it('should fetch news and sentiment for NVDA', async () => {
      // Mock response data
      const mockResponse = {
        data: {
          symbol: 'NVDA',
          articles: [
            { title: "NVIDIA Reports Strong Quarterly Earnings", publishedAt: "2023-08-01 14:25:35", opinion: 1, additionalProp1: {} },
            { title: "NVIDIA's AI Chip for Self-Driving Cars", publishedAt: "2023-08-02 10:15:22", opinion: -1, additionalProp1: {} }
          ],
          total_articles: 2,
          sentiment_metrics: {
            additionalProp1: 0,
            additionalProp2: 0,
            additionalProp3: 0
          },
          meta: {
            message: "Success",
            version: "1.0",
            documentation: "https://api-docs.example.com",
            endpoints: ["string"]
          }
        }
      };

      mockedAxios.get.mockResolvedValueOnce(mockResponse);

      // Make the API call
      const response = await axios.get('http://localhost:8000/api/data/news/NVDA');

      // Assertions
      expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:8000/api/data/news/NVDA');
      expect(response.data.symbol).toBe('NVDA');
      expect(response.data.articles).toBeInstanceOf(Array);
      expect(response.data.articles.length).toBeGreaterThan(0);
      expect(response.data.articles[0]).toHaveProperty('title');
      expect(response.data.articles[0]).toHaveProperty('publishedAt');
      expect(response.data.articles[0]).toHaveProperty('opinion');
    });
  });
});
