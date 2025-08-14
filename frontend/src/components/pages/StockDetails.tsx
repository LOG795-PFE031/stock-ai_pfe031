import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Heading,
  Text,
  Flex,
  Grid,
  GridItem,
  Badge,
  Spinner,
  Alert,
  AlertIcon,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useToast,
  Button
} from '@chakra-ui/react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import PriceAreaChart from '../PriceAreaChart';
import apiService, { StockPrediction, SentimentAnalysis, StockData } from '../../clients/ApiService';
import { getBusinessDateRange } from '../../utils/dateUtils';

const StockDetails: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();
  const toast = useToast();
  const [buttonClickable, setButtonClickable] = useState(false);
  
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState<StockPrediction | null>(null);
  const [sentimentData, setSentimentData] = useState<SentimentAnalysis[]>([]);
  const [historicalData, setHistoricalData] = useState<{date: string, price: number, volume:number}[]>([]);
  const [currentData, setCurrentData] = useState<StockData | null>(null);
  const [predictedData, setPredictedData] = useState<{date: string, price: number}[]>([]);
  const [stockName, setStockName] = useState<string>('');
  const [error, setError] = useState('');
  const [model, setModel] = useState('lstm');
  const [allModelType, setAllModelType] = useState<string[]>([])
  
useEffect(() => {
  const fetchData = async () => {
    if (!ticker) return;
    
    setLoading(true);
    setError('');
    
    try {
      setPredictedData([]);

      // These can run in parallel since they don't depend on each other
      const [modelsResponse, { startDate, endDate }] = await Promise.all([
        apiService.getModelsTypes(),
        getBusinessDateRange(),
      ]);
      
      setAllModelType(modelsResponse.types);

      // Fire off the slow getStockHistoricalPrediction separately 
      apiService.getStockHistoricalPrediction(
        ticker,
        model,
        startDate,
        endDate
      ).then((result) => {
        if (result) {
          const predictionsData = result.predictions.map((pred) => ({
            date: pred.date,
            price: pred.predicted_price
          }));
          
          setPredictedData(predictionsData); // Update state when done
        }
      }).catch((err) => {
        console.error("Historical prediction failed:", err);
      });

      // These API calls can also run in parallel
      const [predictionData, sentimentAnalysis, currentData, historicalData] = await Promise.all([
        apiService.getStockPrediction(ticker, model),
        apiService.getSentimentAnalysis(ticker),
        apiService.getStockData(ticker),
        apiService.getStockDataHistory(ticker, startDate, endDate),
      ]);

      // Update state with all the results
      setPrediction(predictionData);
      setSentimentData(sentimentAnalysis);
      
      const formattedHistoricalData = formatHistoricalData(historicalData.data);
      setHistoricalData(formattedHistoricalData);
      setStockName(currentData.name);
      setCurrentData(currentData.data[0]);
      
    } catch (err) {
      console.error('Error fetching stock data:', err);
      setError('Failed to load stock data. Please try again later.');
      toast({
        title: 'Error',
        description: 'Failed to load stock data.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };
  
  fetchData();
}, [ticker, model, toast]);
  
  // Helper to format the historical data into the correct format
  const formatHistoricalData = (rawHistoricalData: StockData[]) => {
    const data = rawHistoricalData.map(entry => ({
      date: entry.Date.split('T')[0],          // Extract YYYY-MM-DD
      price: entry.Close,                      // Using 'Close' as the price
      volume: entry.Volume,
    }));
    
    return data;
  };

  // Helper to calculate price change and percent change
  const getPriceChange = () => {
    if (historicalData.length < 2) return null;
    const last = historicalData[historicalData.length - 1];
    const prev = historicalData[historicalData.length - 2];
    const change = last.price - prev.price;
    const percent = (change / prev.price) * 100;
    return {
      change: change.toFixed(2),
      percent: percent.toFixed(2),
      isPositive: change >= 0
    };
  };
  
  // Calculate sentiment averages
  const calculateSentimentAverages = () => {
    if (sentimentData.length === 0) return { positive: 0, neutral: 0, negative: 0 };
    
    const totals = sentimentData.reduce(
      (acc, item) => {
        acc.positive += item.sentiment_scores.positive;
        acc.neutral += item.sentiment_scores.neutral;
        acc.negative += item.sentiment_scores.negative;
        return acc;
      },
      { positive: 0, neutral: 0, negative: 0 }
    );
    
    const count = sentimentData.length;
    return {
      positive: totals.positive / count,
      neutral: totals.neutral / count,
      negative: totals.negative / count,
    };
  };
  
  const sentimentAverages = calculateSentimentAverages();
  const dominantSentiment = Object.entries(sentimentAverages).reduce(
    (max, [key, value]) => (value > max.value ? { key, value } : max),
    { key: 'neutral', value: 0 }
  ).key;
  
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'green.500';
      case 'negative':
        return 'red.500';
      default:
        return 'gray.500';
    }
  };


  const updateModelType = async (model_type: string) =>{
    setModel(model_type)
    setLoading(true);
    
    try {
      if (!ticker) throw new Error('Ticker is undefined');
      
      // Check if the model exists before attempting prediction
      const modelExists = await apiService.checkModelExists(ticker, model_type);
      if (!modelExists) {
        console.log(`StockDetails: Model ${model_type}_${ticker} does not exist`);
        setPrediction(null);
        toast({
          title: 'Model Not Available',
          description: `No trained model found for ${ticker} with ${model_type}. Please train the model first.`,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        return;
      }
      
      const dataPredict = await apiService.getStockPrediction(ticker, model_type);
      setPrediction(dataPredict);
    } catch (err) {
      console.error('Error fetching prediction:', err);
      toast({
          title: 'Error',
          description: 'Failed to load stock data.',
          status: 'error',
          duration: 5000,
          isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  }

  const trainData = async () =>{
    try {
      if (!ticker) throw new Error('Ticker is undefined');
      apiService.trainStock(ticker, model).then(async (response) =>{
        if(response == 200 ){
          setLoading(true)
          toast({
          title: 'Training Stock Successful',
          description: 'You have been successfully train the stock',
          status: 'success',
          duration: 3000,
          isClosable: true,
          });

          const predictionData = await apiService.getStockPrediction(ticker, model);
          setPrediction(predictionData);
          setButtonClickable(false)
          setLoading(false)
        }
      })
    } catch (err) {
      console.error('Error fetching prediction:', err);
      toast({
          title: 'Error',
          description: 'Failed to train stock data.',
          status: 'error',
          duration: 5000,
          isClosable: true,
      });
    } 
  }

  
  if (loading) {
    return (
      <Container centerContent py={10}>
        <Spinner size="xl" color="brand.500" />
        <Text mt={4}>Loading stock data...</Text>
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container maxW="container.xl" py={8}>
        <Alert status="error" borderRadius="md">
          <AlertIcon />
          {error}
        </Alert>
      </Container>
    );
  }
  
  return (
    <Container maxW="container.xl" py={8}>
      <Box mb={8}>
        <Flex justify="space-between" align="center" mb={4}>
          <Box>
            <Heading as="h1" size="xl">
              {ticker}
            </Heading>
            <Text fontSize="lg" color="gray.600">
              {stockName || 'Stock Details'}
            </Text>
          </Box>

          <Box>
            <Badge 
              colorScheme={dominantSentiment === 'positive' ? 'green' : dominantSentiment === 'negative' ? 'red' : 'gray'} 
              fontSize="md" 
              p={2} 
              borderRadius="md"
            >
              {dominantSentiment.charAt(0).toUpperCase() + dominantSentiment.slice(1)} Sentiment
            </Badge>
          </Box>
        </Flex>
        
        {/* Price information and prediction */}
        <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6} mb={8}>
          <GridItem>
            <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
              <Heading size="md" mb={4}>Current Price</Heading>
              <Stat>
                <StatLabel fontSize="md">Last traded price</StatLabel>
                <StatNumber fontSize="3xl">
                  ${currentData ? currentData.Close.toFixed(2) : 'N/A'}
                </StatNumber>
                <StatHelpText>
                  {(() => {
                    const priceChange = getPriceChange();
                    if (priceChange) {
                      return (
                        <Text color={priceChange.isPositive ? "green.500" : "red.500"}>
                          {priceChange.isPositive ? "+" : ""}
                          {priceChange.change} ({priceChange.isPositive ? "+" : ""}
                          {priceChange.percent}%) Today
                        </Text>
                      );
                    } else {
                      return <Text color="gray.500">No change data</Text>;
                    }
                  })()}
                </StatHelpText>
              </Stat>
            </Box>
          </GridItem>
          
          <GridItem>
            <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
              <Heading size="md" mb={4}>Price Prediction</Heading>
              {prediction ? (
                <Stat>
                  <StatLabel fontSize="md">Next trading day's predicted price</StatLabel>
                  <StatNumber 
                    fontSize="3xl"
                    color={
                      !currentData || currentData.Close == null
                        ? 'gray.500'
                        : prediction.predicted_price > currentData.Close
                        ? 'green.500'
                        : 'red.500'
                    }
                    >${prediction.predicted_price.toFixed(2)}
                  </StatNumber>
                  <StatHelpText>
                    <Flex align="center" justify="space-between">
                      <Text>Confidence: {(prediction.confidence * 100).toFixed(1)}%</Text>
                    </Flex>
                  </StatHelpText>
                </Stat>
              ) : (
                <Text>No prediction data available</Text>
              )}
              <Flex align="center" justify="space-between">
                { !prediction ? (
                  <Button 
                  size="sm" 
                  colorScheme="brand"
                  disabled = {buttonClickable}
                  onClick={() => { setButtonClickable(true); trainData()}}
                >
                  Train Data
                </Button>

                ): ""

                }
                
                <Text fontSize="sm" color="gray.500">Model:</Text>
                <select value={model} onChange={(type) => updateModelType(type.target.value)} style={{backgroundColor:'beige', width:'150px'}}>
                {allModelType.map((val, index) => (
                  <option key={index} value={val}>
                    {val}
                  </option>
                  ))}
                </select>
              </Flex>
            </Box>
          </GridItem>
        </Grid>
      </Box>
      
      <Tabs variant="soft-rounded" colorScheme="blue" mb={8}>
        <TabList>
          <Tab>Price Chart</Tab>
          <Tab>Sentiment Analysis</Tab>
          <Tab>News Articles</Tab>
        </TabList>
        
        <TabPanels>
          {/* Price Chart */}
          <TabPanel>
            <PriceAreaChart data={historicalData} predictedData={predictedData} />
            {prediction && (
              <Box p={4} borderWidth="1px" borderRadius="md" bg="blue.50">
                <Heading size="sm" mb={2}>Price Prediction Insight</Heading>
                <Text>
                  Based on our {prediction.model_type} model, we predict that {ticker} will be priced at 
                  ${prediction.predicted_price.toFixed(2)} on {new Date(prediction.date).toLocaleDateString()}. 
                  This prediction has a confidence score of {(prediction.confidence * 100).toFixed(1)}%.
                </Text>
              </Box>
            )}
          </TabPanel>
          
          {/* Sentiment Analysis */}
          <TabPanel>
            <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
              <GridItem>
                <Box height="300px">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={[
                        { name: 'Positive', value: sentimentAverages.positive },
                        { name: 'Neutral', value: sentimentAverages.neutral },
                        { name: 'Negative', value: sentimentAverages.negative },
                      ]}
                      margin={{
                        top: 10,
                        right: 30,
                        left: 0,
                        bottom: 0,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#8884d8" 
                        fill="#8884d8" 
                        fillOpacity={0.2} 
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </GridItem>
              
              <GridItem>
                <Box p={5} shadow="md" borderWidth="1px" borderRadius="md" height="100%">
                  <Heading size="md" mb={4}>Sentiment Overview</Heading>
                  
                  {Object.entries(sentimentAverages).map(([key, value]) => (
                    <Box key={key} mb={3}>
                      <Flex justify="space-between" mb={1}>
                        <Text fontWeight="medium">
                          {key.charAt(0).toUpperCase() + key.slice(1)}
                        </Text>
                        <Text>{(value * 100).toFixed(1)}%</Text>
                      </Flex>
                      <Box 
                        w="100%" 
                        bg="gray.100" 
                        h="8px" 
                        borderRadius="full" 
                        overflow="hidden"
                      >
                        <Box 
                          bg={getSentimentColor(key)}
                          h="100%" 
                          w={`${value * 100}%`} 
                          borderRadius="full"
                        />
                      </Box>
                    </Box>
                  ))}
                  
                  <Text mt={4}>
                    Based on {sentimentData.length} news articles, the market sentiment for {ticker} is predominantly {dominantSentiment}.
                  </Text>
                </Box>
              </GridItem>
            </Grid>
          </TabPanel>
          
          {/* News Articles */}
          <TabPanel>
            <Box>
              <Heading size="md" mb={4}>Recent News Articles</Heading>
              
              {sentimentData.length > 0 ? (
                sentimentData.map((article, index) => (
                  <Box key={index} p={4} borderWidth="1px" borderRadius="md" mb={4}>
                    <Heading size="sm" mb={2}>
                      <a href={article.url} target="_blank" rel="noopener noreferrer">
                        {article.title}
                      </a>
                    </Heading>
                    <Text fontSize="sm" color="gray.600" mb={2}>
                      {new Date(article.date).toLocaleDateString()} 
                    </Text>
                    <Flex justify="space-between">
                      <Badge 
                        colorScheme={
                          article.sentiment_scores.positive > article.sentiment_scores.negative ? 'green' : 
                          article.sentiment_scores.negative > article.sentiment_scores.positive ? 'red' : 'gray'
                        }
                      >
                        {
                          article.sentiment_scores.positive > article.sentiment_scores.negative ? 'Positive' : 
                          article.sentiment_scores.negative > article.sentiment_scores.positive ? 'Negative' : 'Neutral'
                        }
                      </Badge>
                      <Box>
                        <Text fontSize="xs" as="span" mr={2}>
                          Positive: {(article.sentiment_scores.positive * 100).toFixed(1)}%
                        </Text>
                        <Text fontSize="xs" as="span" mr={2}>
                          Neutral: {(article.sentiment_scores.neutral * 100).toFixed(1)}%
                        </Text>
                        <Text fontSize="xs" as="span">
                          Negative: {(article.sentiment_scores.negative * 100).toFixed(1)}%
                        </Text>
                      </Box>
                    </Flex>
                  </Box>
                ))
              ) : (
                <Text>No news articles available for {ticker}</Text>
              )}
            </Box>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Container>
  );
};

export default StockDetails; 
