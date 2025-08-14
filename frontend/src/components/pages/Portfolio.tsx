import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Box,
  Container,
  Heading,
  Text,
  Button,
  SimpleGrid,
  Flex,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Input,
  FormControl,
  FormLabel,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Spinner,
  useToast,
  UnorderedList,
  ListItem
} from '@chakra-ui/react';
import { IconPlus, IconTrash, IconChartPie } from '@tabler/icons-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';
import { Methods } from 'openai/resources/fine-tuning/methods.mjs';
import { AuthServer } from '../../clients/AuthServer';

// Define portfolio data types
interface StockHolding {
  id: string;
  symbol: string;
  name: string;
  shares: number;
  purchasePrice: number;
  currentPrice: number;
}

interface ApiShareData {
  symbol: string;
  volume: number;
}

// Colors for the pie chart
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A569BD', '#5DADE2', '#48C9B0'];

// Stock name mapping (ideally this would come from an API)
const STOCK_NAMES: Record<string, string> = {
  'AAPL': 'Apple Inc.',
  'MSFT': 'Microsoft Corp.',
  'GOOGL': 'Alphabet Inc.',
  'GOOG': 'Alphabet Inc.',
  'AMZN': 'Amazon.com Inc.',
  'META': 'Meta Platforms Inc.',
  'TSLA': 'Tesla Inc.',
  'NVDA': 'NVIDIA Corp.',
  'BRK.A': 'Berkshire Hathaway Inc.',
  'BRK.B': 'Berkshire Hathaway Inc.',
  'JPM': 'JPMorgan Chase & Co.',
  'JNJ': 'Johnson & Johnson',
  'V': 'Visa Inc.',
  'PG': 'Procter & Gamble Co.',
  'UNH': 'UnitedHealth Group Inc.',
  'HD': 'Home Depot Inc.',
  'BAC': 'Bank of America Corp.',
  'MA': 'Mastercard Inc.',
  'DIS': 'Walt Disney Co.',
  'ADBE': 'Adobe Inc.'
};

const Portfolio: React.FC = () => {
  const navigate = useNavigate();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  
  // State for portfolio data
  const [holdings, setHoldings] = useState<StockHolding[]>([]);
  const [loading, setLoading] = useState(true);
  const [walletId, setWalletId] = useState('true');
  const [error, setError] = useState<string | null>(null);
  
  const [newStock, setNewStock] = useState<Partial<StockHolding>>({
    symbol: '',
    name: '',
    shares: 0,
    purchasePrice: 0
  });

  const authServer = new AuthServer(import.meta.env.VITE_API_AUTH_URL || 'https://localhost:55604');

  // Fetch portfolio data
  useEffect(() => {
    fetchPortfolioData();
  }, []);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      // const portfolioApiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || 'https://localhost:55616';
      // const stocksApiUrl = import.meta.env.VITE_API_STOCKS_URL || 'https://localhost:55611';
      
      // try {
        // Get user's portfolio (shares)
        // console.log('dans le bhay portfols')

        const walletIdResponse = await authServer.getUserWalletId() 
        setWalletId(walletIdResponse.data.value)
        const myStock: StockHolding = {
          id: '1',
          symbol: 'AAPL',
          name: 'Apple Inc.',
          shares: 10,
          purchasePrice: 150.00,
          currentPrice: 180.50
        };
        const tab = []
        tab.push(myStock)
        setHoldings(tab)
        // const sharesResponse = await authServer.getUserShares(walletIdResponse.data.value)
        // console.log(sharesResponse)

        // console.log(portfolioApiUrl)
        // const portfolioResponse = await fetch(`${portfolioApiUrl}/portfolio`,{
        //   method: 'GET',
        //   headers: {
        //     'Content-Type': 'application/json',
        //   },
        // });

      //   console.log('dans le bhay portfols')
      //   const portfolioResponse = await axios.get(`${portfolioApiUrl}/portfolio`);
      //   console.log(portfolioResponse)
        
      //   console.log('dans le bhay portfols')
      //   if (!portfolioResponse.data || !portfolioResponse.data.shareVolumes) {
      //     // Handle empty but valid response
      //     setHoldings([]);
      //     setLoading(false);
      //     return;
      //   }
        
      //   const shares: ApiShareData[] = portfolioResponse.data.shareVolumes;
        
      //   // Create StockHolding objects with placeholder data
      //   const processedHoldings: StockHolding[] = [];
        
      //   // // Get current price for each stock and calculate metrics
      //   for (const share of shares) {
      //     try {
      //       // Get current price from stocks API
      //       const live = new Date().toISOString();
      //       const priceResponse = await axios.patch(`${stocksApiUrl}/stocks/${share.symbol}/${live}`);
            
      //       const currentPrice = priceResponse.data?.value || 0;
            
      //       // For now, we don't have purchase price data from the API, so we'll estimate it
      //       // In a real app, this should come from transaction history
      //       // We'll estimate it as 80-90% of current price for demonstration
      //       const estimateFactor = 0.8 + (Math.random() * 0.1); // 80-90% of current price
      //       const purchasePrice = currentPrice * estimateFactor;
            
      //       // Add to holdings
      //       processedHoldings.push({
      //         id: crypto.randomUUID(),
      //         symbol: share.symbol,
      //         name: STOCK_NAMES[share.symbol] || `${share.symbol} Stock`,
      //         shares: share.volume,
      //         purchasePrice: parseFloat(purchasePrice.toFixed(2)),
      //         currentPrice: parseFloat(currentPrice.toFixed(2))
      //       });
      //     } catch (err) {
      //       console.error(`Error fetching price for ${share.symbol}:`, err);
      //       // Still add the stock but with placeholder price data
      //       processedHoldings.push({
      //         id: crypto.randomUUID(),
      //         symbol: share.symbol,
      //         name: STOCK_NAMES[share.symbol] || `${share.symbol} Stock`,
      //         shares: share.volume,
      //         purchasePrice: 0,
      //         currentPrice: 0
      //       });
      //     }
      //   }
        
      //   setHoldings(processedHoldings);
      // } catch (err) {
      //   // Check if this is a 404 error (portfolio not found or empty)
      //   if (axios.isAxiosError(err) && err.response?.status === 404) {
      //     // This is not really an error - just an empty portfolio
      //     setHoldings([]);
      //   } else {
      //     // This is a real error
      //     throw err;
      //   }
      // }
    } catch (err) {
      console.error('Error fetching portfolio data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio data');
    } finally {
      setLoading(false);
    }
  };
  
  // Calculate portfolio value and performance metrics
  const calculatePortfolioMetrics = () => {
    let totalValue = 0;
    let totalCost = 0;
    
    holdings.forEach(stock => {
      totalValue += stock.currentPrice * stock.shares;
      totalCost += stock.purchasePrice * stock.shares;
    });
    
    const totalGainLoss = totalValue - totalCost;
    const percentageGainLoss = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;
    
    return {
      totalValue,
      totalCost,
      totalGainLoss,
      percentageGainLoss
    };
  };
  
  const metrics = calculatePortfolioMetrics();
  
  // Prepare data for pie chart
  const pieChartData = holdings.map(stock => ({
    name: stock.symbol,
    value: stock.currentPrice * stock.shares
  }));
  
  // Handle adding a new stock
  const handleAddStock = async () => {
    if (!newStock.symbol || !newStock.shares) {
      toast({
        title: "Missing information",
        description: "Please enter both symbol and number of shares",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    try {
      const apiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || 'https://localhost:55616';
      // await axios.patch(`${apiUrl}/portfolio/buy/${newStock.symbol}/${newStock.shares}/${walletId}`);
      console.log(newStock.symbol)
      console.log(newStock.shares)
      await axios.patch(`${apiUrl}/portfolio/buy/${newStock.symbol}/${newStock.shares}/${walletId}`);
      
      toast({
        title: "Stock purchased",
        description: `Successfully added ${newStock.shares} shares of ${newStock.symbol}`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      
      // Refresh data
      fetchPortfolioData();
      
      // Reset form
      setNewStock({
        symbol: '',
        name: '',
        shares: 0,
        purchasePrice: 0
      });
      
      onClose();
    } catch (err) {
      console.error('Error adding stock:', err);
      toast({
        title: "Purchase failed",
        description: err instanceof Error ? err.message : "An error occurred while adding the stock",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };
  
  // Handle removing a stock
  const handleRemoveStock = async (stock: StockHolding) => {
    try {
      const apiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || 'https://localhost:55616';
      await axios.patch(`${apiUrl}/portfolio/sell/${stock.symbol}/${stock.shares}`);
      
      toast({
        title: "Stock sold",
        description: `Successfully sold all shares of ${stock.symbol}`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      
      // Refresh data
      fetchPortfolioData();
    } catch (err) {
      console.error('Error removing stock:', err);
      toast({
        title: "Sale failed",
        description: err instanceof Error ? err.message : "An error occurred while selling the stock",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };
  
  // Handle viewing a stock
  const handleViewStock = (symbol: string) => {
    navigate(`/stock/${symbol}`);
  };
  
  // Calculate individual stock performance
  const calculateStockPerformance = (stock: StockHolding) => {
    const value = stock.currentPrice * stock.shares;
    const cost = stock.purchasePrice * stock.shares;
    const gainLoss = value - cost;
    const percentageGainLoss = cost > 0 ? (gainLoss / cost) * 100 : 0;
    
    return {
      value,
      gainLoss,
      percentageGainLoss
    };
  };

  // Show loading state
  if (loading) {
    return (
      <Container maxW="container.xl" py={8}>
        <Flex direction="column" align="center" justify="center" py={10}>
          <Spinner size="xl" mb={4} color="blue.500" />
          <Text fontSize="lg">Loading your portfolio...</Text>
        </Flex>
      </Container>
    );
  }

  // Show a more helpful error state
  if (error) {
    return (
      <Container maxW="container.xl" py={8}>
        <Box p={6} borderWidth="1px" borderRadius="lg" bg="red.50">
          <Heading size="md" color="red.500" mb={3}>Error loading portfolio</Heading>
          <Text mb={4}>{error}</Text>
          <Text mb={4}>
            This could be due to a network issue, server problem, or authentication error. 
            Here are some steps you can try:
          </Text>
          <UnorderedList mb={6} pl={4}>
            <ListItem mb={2}>Verify your internet connection</ListItem>
            <ListItem mb={2}>Try refreshing the page</ListItem>
            <ListItem mb={2}>Log out and log back in</ListItem>
            <ListItem>Try again in a few moments</ListItem>
          </UnorderedList>
          <Flex gap={4}>
            <Button colorScheme="blue" onClick={fetchPortfolioData}>
              Retry
            </Button>
            <Button variant="outline" onClick={onOpen}>
              Add Stock Anyway
            </Button>
          </Flex>
        </Box>

        {/* Add Stock Modal */}
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Add Stock to Portfolio</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <FormControl mb={4}>
                <FormLabel>Symbol</FormLabel>
                <Input 
                  value={newStock.symbol}
                  onChange={(e) => setNewStock({...newStock, symbol: e.target.value.toUpperCase()})}
                  placeholder="e.g. AAPL"
                />
                <Text fontSize="sm" color="gray.500" mt={1}>
                  Try AAPL (Apple), MSFT (Microsoft), GOOGL (Google), or AMZN (Amazon)
                </Text>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Number of Shares</FormLabel>
                <Input 
                  type="number"
                  value={newStock.shares || ''}
                  onChange={(e) => setNewStock({...newStock, shares: parseInt(e.target.value)})}
                  placeholder="e.g. 10"
                />
              </FormControl>
            </ModalBody>

            <ModalFooter>
              <Button colorScheme="gray" mr={3} onClick={onClose}>
                Cancel
              </Button>
              <Button 
                colorScheme="blue" 
                onClick={handleAddStock}
                isDisabled={!newStock.symbol || !newStock.shares}
                isLoading={loading}
              >
                Add to Portfolio
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Container>
    );
  }

  // Show empty state
  if (holdings.length === 0) {
    return (
      <Container maxW="container.xl" py={8}>
        <Flex justify="space-between" align="center" mb={8}>
          <Box>
            <Heading as="h1" size="xl" mb={2}>Your Portfolio</Heading>
            <Text color="gray.600">Start building your investment portfolio</Text>
          </Box>
          <Button 
            leftIcon={<IconPlus size={16} />} 
            colorScheme="blue" 
            onClick={onOpen}
          >
            Add Stock
          </Button>
        </Flex>
        
        <Box p={10} borderWidth="1px" borderRadius="lg" textAlign="center">
          <Heading size="md" mb={4}>Welcome to Your Investment Journey</Heading>
          <Text mb={6}>
            You don't have any stocks in your portfolio yet. Start by adding some stocks
            to track their performance and build your wealth.
          </Text>
          
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6} mt={8} mb={8}>
            <Box p={5} borderWidth="1px" borderRadius="md" bg="blue.50">
              <Heading size="md" mb={3} color="blue.600">1. Add Stocks</Heading>
              <Text>
                Click the "Add Stock" button to purchase your first stock. Choose from
                popular companies like Apple, Microsoft, Google, and more.
              </Text>
            </Box>
            
            <Box p={5} borderWidth="1px" borderRadius="md" bg="green.50">
              <Heading size="md" mb={3} color="green.600">2. Track Performance</Heading>
              <Text>
                Once you've added stocks, you can track their performance, see your gains,
                and monitor your overall portfolio value.
              </Text>
            </Box>
            
            <Box p={5} borderWidth="1px" borderRadius="md" bg="purple.50">
              <Heading size="md" mb={3} color="purple.600">3. Grow Your Portfolio</Heading>
              <Text>
                Continue building your portfolio by adding more stocks or increasing your
                positions in existing ones.
              </Text>
            </Box>
          </SimpleGrid>
          
          <Button 
            leftIcon={<IconPlus size={16} />} 
            colorScheme="blue" 
            size="lg"
            onClick={onOpen}
          >
            Add Your First Stock
          </Button>
        </Box>
        
        {/* Add Stock Modal */}
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>Add Stock to Portfolio</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <FormControl mb={4}>
                <FormLabel>Symbol</FormLabel>
                <Input 
                  value={newStock.symbol}
                  onChange={(e) => setNewStock({...newStock, symbol: e.target.value.toUpperCase()})}
                  placeholder="e.g. AAPL"
                />
                <Text fontSize="sm" color="gray.500" mt={1}>
                  Try AAPL (Apple), MSFT (Microsoft), GOOGL (Google), or AMZN (Amazon)
                </Text>
              </FormControl>
              
              <FormControl mb={4}>
                <FormLabel>Number of Shares</FormLabel>
                <Input 
                  type="number"
                  value={newStock.shares || ''}
                  onChange={(e) => setNewStock({...newStock, shares: parseInt(e.target.value)})}
                  placeholder="e.g. 10"
                />
              </FormControl>
            </ModalBody>

            <ModalFooter>
              <Button colorScheme="gray" mr={3} onClick={onClose}>
                Cancel
              </Button>
              <Button 
                colorScheme="blue" 
                onClick={handleAddStock}
                isDisabled={!newStock.symbol || !newStock.shares}
                isLoading={loading}
              >
                Add to Portfolio
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Container>
    );
  }
  
  return (
    <Container maxW="container.xl" py={8}>
      <Flex justify="space-between" align="center" mb={8}>
        <Box>
          <Heading as="h1" size="xl" mb={2}>Your Portfolio</Heading>
          <Text color="gray.600">Track and manage your investments</Text>
        </Box>
        <Button 
          leftIcon={<IconPlus size={16} />} 
          colorScheme="blue" 
          onClick={onOpen}
        >
          Add Stock
        </Button>
      </Flex>
      
      {/* Portfolio Overview */}
      <Box mb={10}>
        <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6} mb={6}>
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
            <Stat>
              <StatLabel>Total Value</StatLabel>
              <StatNumber>${metrics.totalValue.toFixed(2)}</StatNumber>
              <StatHelpText>
                <StatArrow type={metrics.totalGainLoss >= 0 ? 'increase' : 'decrease'} />
                ${Math.abs(metrics.totalGainLoss).toFixed(2)} ({metrics.percentageGainLoss.toFixed(2)}%)
              </StatHelpText>
            </Stat>
          </Box>
          
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
            <Stat>
              <StatLabel>Total Cost</StatLabel>
              <StatNumber>${metrics.totalCost.toFixed(2)}</StatNumber>
            </Stat>
          </Box>
          
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
            <Stat>
              <StatLabel>Total Gain/Loss</StatLabel>
              <StatNumber color={metrics.totalGainLoss >= 0 ? 'green.500' : 'red.500'}>
                ${metrics.totalGainLoss.toFixed(2)}
              </StatNumber>
            </Stat>
          </Box>
          
          <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
            <Stat>
              <StatLabel>Performance</StatLabel>
              <StatNumber color={metrics.percentageGainLoss >= 0 ? 'green.500' : 'red.500'}>
                {metrics.percentageGainLoss.toFixed(2)}%
              </StatNumber>
              <StatHelpText>Since purchase</StatHelpText>
            </Stat>
          </Box>
        </SimpleGrid>
        
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
          <Box shadow="md" borderWidth="1px" borderRadius="md" p={5}>
            <Heading size="md" mb={4}>Portfolio Allocation</Heading>
            <Box height="300px">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieChartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [
                    `$${typeof value === 'number' ? value.toFixed(2) : Number(value).toFixed(2)}`, 
                    'Value'
                  ]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Box>
          
          <Box shadow="md" borderWidth="1px" borderRadius="md" p={5}>
            <Heading size="md" mb={4}>Performance Overview</Heading>
            <Box overflowY="auto" maxH="300px">
              {holdings.map(stock => {
                const performance = calculateStockPerformance(stock);
                const percentWidth = Math.min(Math.abs(performance.percentageGainLoss), 100);
                
                return (
                  <Box key={stock.id} mb={4}>
                    <Flex justify="space-between" align="center" mb={1}>
                      <Text fontWeight="bold">{stock.symbol}</Text>
                      <Text 
                        fontWeight="medium" 
                        color={performance.percentageGainLoss >= 0 ? 'green.500' : 'red.500'}
                      >
                        {performance.percentageGainLoss.toFixed(2)}%
                      </Text>
                    </Flex>
                    <Progress 
                      value={percentWidth} 
                      colorScheme={performance.percentageGainLoss >= 0 ? 'green' : 'red'} 
                      size="sm" 
                      borderRadius="full"
                    />
                  </Box>
                );
              })}
            </Box>
          </Box>
        </SimpleGrid>
      </Box>
      
      {/* Holdings Table */}
      <Box shadow="md" borderWidth="1px" borderRadius="md" overflow="hidden">
        <Heading size="md" p={5} borderBottomWidth="1px">
          Your Holdings
        </Heading>
        
        <Box overflowX="auto">
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>Symbol</Th>
                <Th>Name</Th>
                <Th isNumeric>Shares</Th>
                <Th isNumeric>Purchase Price</Th>
                <Th isNumeric>Current Price</Th>
                <Th isNumeric>Total Value</Th>
                <Th isNumeric>Gain/Loss</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {holdings.map(stock => {
                const performance = calculateStockPerformance(stock);
                
                return (
                  <Tr key={stock.id}>
                    <Td fontWeight="medium">{stock.symbol}</Td>
                    <Td>{stock.name}</Td>
                    <Td isNumeric>{stock.shares}</Td>
                    <Td isNumeric>${stock.purchasePrice.toFixed(2)}</Td>
                    <Td isNumeric>${stock.currentPrice.toFixed(2)}</Td>
                    <Td isNumeric>${performance.value.toFixed(2)}</Td>
                    <Td isNumeric color={performance.gainLoss >= 0 ? 'green.500' : 'red.500'}>
                      ${performance.gainLoss.toFixed(2)} ({performance.percentageGainLoss.toFixed(2)}%)
                    </Td>
                    <Td>
                      <Flex>
                        <Button 
                          size="sm" 
                          colorScheme="blue" 
                          variant="ghost" 
                          onClick={() => handleViewStock(stock.symbol)}
                          mr={2}
                        >
                          <IconChartPie size={18} />
                        </Button>
                        <Button 
                          size="sm" 
                          colorScheme="red" 
                          variant="ghost" 
                          onClick={() => handleRemoveStock(stock)}
                        >
                          <IconTrash size={18} />
                        </Button>
                      </Flex>
                    </Td>
                  </Tr>
                );
              })}
            </Tbody>
          </Table>
        </Box>
      </Box>
      
      {/* Add Stock Modal */}
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Add Stock to Portfolio</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <FormControl mb={4}>
              <FormLabel>Symbol</FormLabel>
              <Input 
                value={newStock.symbol}
                onChange={(e) => setNewStock({...newStock, symbol: e.target.value.toUpperCase()})}
                placeholder="e.g. AAPL"
              />
              <Text fontSize="sm" color="gray.500" mt={1}>
                Try AAPL (Apple), MSFT (Microsoft), GOOGL (Google), or AMZN (Amazon)
              </Text>
            </FormControl>
            
            <FormControl mb={4}>
              <FormLabel>Number of Shares</FormLabel>
              <Input 
                type="number"
                value={newStock.shares || ''}
                onChange={(e) => setNewStock({...newStock, shares: parseInt(e.target.value)})}
                placeholder="e.g. 10"
              />
            </FormControl>
          </ModalBody>

          <ModalFooter>
            <Button colorScheme="gray" mr={3} onClick={onClose}>
              Cancel
            </Button>
            <Button 
              colorScheme="blue" 
              onClick={handleAddStock}
              isDisabled={!newStock.symbol || !newStock.shares}
              isLoading={loading}
            >
              Add to Portfolio
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Container>
  );
};

export default Portfolio; 