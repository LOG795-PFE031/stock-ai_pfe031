import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  SimpleGrid,
  Flex,
  Select,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useColorModeValue,
  Icon,
  Button,
  Badge,
  HStack,
  Tooltip as ChakraTooltip
} from '@chakra-ui/react';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Scatter
} from 'recharts';
import { IconTrendingUp, IconTrendingDown, IconCalendar, IconCurrencyDollar } from '@tabler/icons-react';
import apiService from '../../clients/ApiService';

// Types
interface PortfolioValue {
  date: string;
  value: number;
  change: number;
}

interface SectorAllocation {
  name: string;
  value: number;
}

interface StockPerformance {
  symbol: string;
  name: string;
  percentChange: number;
  value: number;
}

const PortfolioPerformance: React.FC = () => {
  const [timeRange, setTimeRange] = useState<string>('1m');
  const [portfolioData, setPortfolioData] = useState<PortfolioValue[]>([]);
  const [sectorAllocation, setSectorAllocation] = useState<SectorAllocation[]>([]);
  const [stockPerformance, setStockPerformance] = useState<StockPerformance[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  
  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#4BC0C0', '#9966FF', '#FF6384'];
  const areaGradientStart = useColorModeValue('rgba(0, 112, 243, 0.2)', 'rgba(0, 112, 243, 0.1)');
  const areaGradientEnd = useColorModeValue('rgba(0, 112, 243, 0.01)', 'rgba(0, 112, 243, 0.01)');
  
  // Portfolio metrics
  const [metrics, setMetrics] = useState({
    totalValue: 0,
    dayChange: 0,
    dayChangePercent: 0,
    allTimeReturn: 0,
    allTimeReturnPercent: 0
  });

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  // Fetch portfolio data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // In a real app, you'd fetch this data from your API
        // For now, we'll generate sample data
        
        // Generate historical portfolio value data
        const generatePortfolioData = (days: number, startValue: number) => {
          const data: PortfolioValue[] = [];
          let currentValue = startValue;
          
          for (let i = days; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            
            // Random daily fluctuation between -3% and +3%
            const fluctuation = currentValue * (Math.random() * 0.06 - 0.03);
            currentValue += fluctuation;
            
            data.push({
              date: date.toISOString().split('T')[0],
              value: currentValue,
              change: fluctuation
            });
          }
          
          return data;
        };
        
        // Generate sector allocation data
        const sectorData: SectorAllocation[] = [
          { name: 'Technology', value: 35 },
          { name: 'Financial', value: 25 },
          { name: 'Healthcare', value: 15 },
          { name: 'Consumer', value: 10 },
          { name: 'Energy', value: 8 },
          { name: 'Industrial', value: 7 }
        ];
        
        // Generate stock performance data
        const stockData: StockPerformance[] = [
          { symbol: 'AAPL', name: 'Apple Inc.', percentChange: 2.5, value: 15000 },
          { symbol: 'MSFT', name: 'Microsoft Corp.', percentChange: 1.7, value: 12000 },
          { symbol: 'GOOGL', name: 'Alphabet Inc.', percentChange: -0.8, value: 9000 },
          { symbol: 'AMZN', name: 'Amazon.com Inc.', percentChange: 3.2, value: 8500 },
          { symbol: 'TSLA', name: 'Tesla Inc.', percentChange: -1.5, value: 7500 }
        ];
        
        // Set days based on timeRange
        let days = 30;
        switch (timeRange) {
          case '1w': days = 7; break;
          case '1m': days = 30; break;
          case '3m': days = 90; break;
          case '1y': days = 365; break;
          case '5y': days = 1825; break;
          default: days = 30;
        }
        
        const portfolioHistory = generatePortfolioData(days, 100000);
        
        // Calculate metrics
        const latestValue = portfolioHistory[portfolioHistory.length - 1].value;
        const previousDayValue = portfolioHistory[portfolioHistory.length - 2].value;
        const initialValue = portfolioHistory[0].value;
        
        const dayChange = latestValue - previousDayValue;
        const allTimeReturn = latestValue - initialValue;
        
        setMetrics({
          totalValue: latestValue,
          dayChange,
          dayChangePercent: (dayChange / previousDayValue) * 100,
          allTimeReturn,
          allTimeReturnPercent: (allTimeReturn / initialValue) * 100
        });
        
        setPortfolioData(portfolioHistory);
        setSectorAllocation(sectorData);
        setStockPerformance(stockData);
      } catch (error) {
        console.error('Error fetching portfolio performance data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [timeRange]);

  // Custom tooltip for the area chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box bg="white" p={3} boxShadow="md" borderRadius="md">
          <Text fontWeight="bold">{label}</Text>
          <Text color="blue.500">
            {formatCurrency(payload[0].value)}
          </Text>
          <Text color={payload[0].payload.change >= 0 ? "green.500" : "red.500"}>
            {payload[0].payload.change >= 0 ? "+" : ""}
            {formatCurrency(payload[0].payload.change)} 
            ({((payload[0].payload.change / (payload[0].value - payload[0].payload.change)) * 100).toFixed(2)}%)
          </Text>
        </Box>
      );
    }
    return null;
  };

  // Show a loading state
  if (loading) {
    return (
      <Container centerContent py={10}>
        <Heading size="md" mb={4}>Loading Portfolio Data...</Heading>
      </Container>
    );
  }

  return (
    <Container maxW="container.xl" py={6}>
      <Heading as="h1" mb={6}>Portfolio Performance</Heading>
      
      {/* Time Range Selector */}
      <Flex justifyContent="space-between" alignItems="center" mb={6}>
        <Text fontSize="lg" fontWeight="medium">
          Total Portfolio Value: {formatCurrency(metrics.totalValue)}
        </Text>
        <HStack>
          <Icon as={IconCalendar} />
          <Select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
            width="auto"
          >
            <option value="1w">1 Week</option>
            <option value="1m">1 Month</option>
            <option value="3m">3 Months</option>
            <option value="1y">1 Year</option>
            <option value="5y">5 Years</option>
          </Select>
        </HStack>
      </Flex>
      
      {/* Portfolio Metrics Cards */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Box p={5} borderRadius="lg" boxShadow="md" bg={useColorModeValue('white', 'gray.700')}>
          <Stat>
            <StatLabel fontSize="sm">Today's Change</StatLabel>
            <Flex alignItems="center">
              <StatNumber>{formatCurrency(metrics.dayChange)}</StatNumber>
              <StatHelpText ml={2} mb={0}>
                <StatArrow type={metrics.dayChange >= 0 ? 'increase' : 'decrease'} />
                {Math.abs(metrics.dayChangePercent).toFixed(2)}%
              </StatHelpText>
            </Flex>
          </Stat>
        </Box>
        
        <Box p={5} borderRadius="lg" boxShadow="md" bg={useColorModeValue('white', 'gray.700')}>
          <Stat>
            <StatLabel fontSize="sm">Total Return</StatLabel>
            <Flex alignItems="center">
              <StatNumber>{formatCurrency(metrics.allTimeReturn)}</StatNumber>
              <StatHelpText ml={2} mb={0}>
                <StatArrow type={metrics.allTimeReturn >= 0 ? 'increase' : 'decrease'} />
                {Math.abs(metrics.allTimeReturnPercent).toFixed(2)}%
              </StatHelpText>
            </Flex>
          </Stat>
        </Box>
        
        <Box p={5} borderRadius="lg" boxShadow="md" bg={useColorModeValue('white', 'gray.700')}>
          <Stat>
            <StatLabel fontSize="sm">Highest Value Stock</StatLabel>
            <StatNumber fontSize="xl">
              {stockPerformance.length > 0 && stockPerformance.sort((a, b) => b.value - a.value)[0].symbol}
            </StatNumber>
            <StatHelpText>
              {stockPerformance.length > 0 && formatCurrency(stockPerformance.sort((a, b) => b.value - a.value)[0].value)}
            </StatHelpText>
          </Stat>
        </Box>
        
        <Box p={5} borderRadius="lg" boxShadow="md" bg={useColorModeValue('white', 'gray.700')}>
          <Stat>
            <StatLabel fontSize="sm">Best Performer</StatLabel>
            <StatNumber fontSize="xl">
              {stockPerformance.length > 0 && stockPerformance.sort((a, b) => b.percentChange - a.percentChange)[0].symbol}
            </StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              {stockPerformance.length > 0 && stockPerformance.sort((a, b) => b.percentChange - a.percentChange)[0].percentChange.toFixed(2)}%
            </StatHelpText>
          </Stat>
        </Box>
      </SimpleGrid>
      
      {/* Main Performance Chart */}
      <Box 
        mb={8} 
        p={6} 
        borderRadius="lg" 
        boxShadow="md" 
        bg={useColorModeValue('white', 'gray.700')}
      >
        <Heading size="md" mb={4}>Portfolio Value Over Time</Heading>
        <Box height="400px">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={portfolioData}
              margin={{
                top: 10,
                right: 30,
                left: 20,
                bottom: 0,
              }}
            >
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0070F3" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#0070F3" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={useColorModeValue('#f0f0f0', '#2D3748')} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }} 
                tickMargin={10}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return new Intl.DateTimeFormat('en-US', {
                    month: 'short',
                    day: 'numeric',
                  }).format(date);
                }}
              />
              <YAxis 
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                domain={['dataMin - 5000', 'dataMax + 5000']}
                tick={{ fontSize: 12 }}
                tickMargin={10}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#0070F3" 
                fill="url(#colorValue)" 
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>
      </Box>
      
      {/* Additional Portfolio Insights */}
      <Tabs variant="enclosed" colorScheme="blue" isLazy>
        <TabList>
          <Tab>Allocation</Tab>
          <Tab>Top Holdings</Tab>
          <Tab>Performance</Tab>
        </TabList>
        
        <TabPanels>
          {/* Sector Allocation Tab */}
          <TabPanel>
            <Box 
              p={6} 
              borderRadius="lg" 
              boxShadow="md" 
              bg={useColorModeValue('white', 'gray.700')}
            >
              <Heading size="md" mb={6}>Portfolio Allocation by Sector</Heading>
              <Flex direction={{ base: 'column', md: 'row' }}>
                <Box width={{ base: '100%', md: '50%' }} height="400px">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={sectorAllocation}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={150}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {sectorAllocation.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => [`${value}%`, 'Allocation']} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
                
                <Box width={{ base: '100%', md: '50%' }} mt={{ base: 6, md: 0 }}>
                  <Heading size="sm" mb={4}>Sector Distribution</Heading>
                  {sectorAllocation.map((sector, index) => (
                    <Flex key={index} mb={3} align="center">
                      <Box 
                        width="16px" 
                        height="16px" 
                        bg={COLORS[index % COLORS.length]} 
                        borderRadius="sm" 
                        mr={3} 
                      />
                      <Text flex="1">{sector.name}</Text>
                      <Text fontWeight="bold">{sector.value}%</Text>
                    </Flex>
                  ))}
                  <Text mt={6} fontSize="sm" color="gray.500">
                    Diversification helps reduce risk. Consider rebalancing if any sector exceeds 40% of your portfolio.
                  </Text>
                </Box>
              </Flex>
            </Box>
          </TabPanel>
          
          {/* Top Holdings Tab */}
          <TabPanel>
            <Box 
              p={6} 
              borderRadius="lg" 
              boxShadow="md" 
              bg={useColorModeValue('white', 'gray.700')}
            >
              <Heading size="md" mb={6}>Top Holdings by Value</Heading>
              <Box height="400px">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={stockPerformance.sort((a, b) => b.value - a.value)}
                    layout="vertical"
                    margin={{
                      top: 20,
                      right: 30,
                      left: 60,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                    <XAxis type="number" tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                    <YAxis dataKey="symbol" type="category" width={80} />
                    <Tooltip 
                      formatter={(value) => [formatCurrency(Number(value)), 'Value']}
                      labelFormatter={(value) => stockPerformance.find(item => item.symbol === value)?.name || value}
                    />
                    <Bar 
                      dataKey="value" 
                      fill="#0070F3"
                      radius={[0, 4, 4, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
              
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4} mt={6}>
                {stockPerformance.sort((a, b) => b.value - a.value).slice(0, 4).map((stock, index) => (
                  <Box key={index} p={3} borderWidth="1px" borderRadius="md">
                    <Flex justify="space-between" align="center">
                      <Box>
                        <Text fontWeight="bold">{stock.symbol}</Text>
                        <Text fontSize="sm" color="gray.500">{stock.name}</Text>
                      </Box>
                      <Stat textAlign="right">
                        <StatNumber fontSize="md">{formatCurrency(stock.value)}</StatNumber>
                        <StatHelpText mb={0}>
                          <StatArrow type={stock.percentChange >= 0 ? 'increase' : 'decrease'} />
                          {Math.abs(stock.percentChange).toFixed(2)}%
                        </StatHelpText>
                      </Stat>
                    </Flex>
                  </Box>
                ))}
              </SimpleGrid>
            </Box>
          </TabPanel>
          
          {/* Performance Comparison Tab */}
          <TabPanel>
            <Box 
              p={6} 
              borderRadius="lg" 
              boxShadow="md" 
              bg={useColorModeValue('white', 'gray.700')}
            >
              <Heading size="md" mb={6}>Stock Performance Comparison</Heading>
              <Box height="400px">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={stockPerformance}
                    margin={{
                      top: 20,
                      right: 30,
                      left: 20,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="symbol" />
                    <YAxis yAxisId="left" label={{ value: 'Value ($)', angle: -90, position: 'insideLeft' }} />
                    <YAxis yAxisId="right" orientation="right" label={{ value: 'Change (%)', angle: 90, position: 'insideRight' }} />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="value" fill="#0088FE" name="Value ($)" />
                    <Line yAxisId="right" type="monotone" dataKey="percentChange" stroke="#FF8042" name="Performance (%)" />
                    <Scatter yAxisId="right" dataKey="percentChange" fill="#FF8042" name="Performance (%)" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Box>
              
              <Text mt={4} fontSize="sm" color="gray.500" textAlign="center">
                This chart shows the relationship between stock value in your portfolio and its performance.
                Larger bubbles represent higher value stocks.
              </Text>
            </Box>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Container>
  );
};

export default PortfolioPerformance; 