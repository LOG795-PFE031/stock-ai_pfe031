import React, { useEffect, useState } from 'react';
import StockHistoryChart from '../models/StockHistoryChart';
import axios from 'axios';

type StockDataPoint = {
  date: string;
  close: number;
  volume: number;
};

type params = {
  searchTerm: string;
  isLoggedIn: boolean
};

const Stock: React.FC<params> = ({ searchTerm, isLoggedIn }) => {
  const [stockData, setStockData] = useState<StockDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoggedIn || !searchTerm) return;
    
    const fetchData = async () => {
      try {
        setLoading(true);
        const apiUrl = import.meta.env.VITE_API_STOCKS_URL || 'https://localhost:55611';
        const live = new Date().toISOString();
        const response = await axios.get(`${apiUrl}/stocks/${searchTerm}/${live}`);

        if (response.data && response.data.value) {
          const formattedData = [{
            date: live, 
            close: response.data.value, 
            volume: 0
          }];
          setStockData(formattedData);
        } else {
          throw new Error('Invalid response format from API');
        }
      } catch (err) {
        console.error('Error fetching stock data:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch stock data');
        setStockData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [searchTerm, isLoggedIn]);

  if (!isLoggedIn) return <div>Please log in to see the stocks.</div>;
  if (loading) return <div>Loading stock data...</div>;
  if (error) return <div>Error: {error}</div>;
  if (stockData.length === 0) return <div>No data available for {searchTerm}</div>;

  return (
    <div style={{ padding: '20px' }}>
      <StockHistoryChart data={stockData} />
    </div>
  );
};

export default Stock;