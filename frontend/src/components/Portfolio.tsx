import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Share from '../models/Share';
import { Box, Button, FormControl, FormLabel, Input, Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton, useDisclosure, Flex, Heading, useToast } from '@chakra-ui/react';

type ShareData = {
    symbol: string;
    volume: number;
}

type params = {
  setSearchTermFinal: React.Dispatch<React.SetStateAction<string>>;
  setSearchTerm: React.Dispatch<React.SetStateAction<string>>;
  setCurrentPage: React.Dispatch<React.SetStateAction<string>>;
  searchTerm: string;
  isLoggedIn: boolean

};

const Portfolio: React.FC<params> = ({ setSearchTermFinal,setSearchTerm,setCurrentPage,searchTerm,isLoggedIn }) => {
    const [shares, setShares] = useState<ShareData[]>([]);
    const [totalShares, setTotalShares]  = useState<number>(0);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedSymbol, setSelectedSymbol] = useState<string>('');
    const [transactionVolume, setTransactionVolume] = useState<number>(0);
    const toast = useToast();

    // Buy and sell modals
    const { isOpen: isBuyOpen, onOpen: onBuyOpen, onClose: onBuyClose } = useDisclosure();
    const { isOpen: isSellOpen, onOpen: onSellOpen, onClose: onSellClose } = useDisclosure();

    const fetchData = async () => {
        try {
          setLoading(true);
          setError(null); // Reset error before fetch
          const apiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || "https://localhost:55616";
          
          try {
            const response = await axios.get(`${apiUrl}/portfolio`);
            
            if (response.data && response.data.shareVolumes) {
          setShares(response.data.shareVolumes);
              setTotalShares(0); // Reset total, will be calculated with share prices
            } else {
              // If we get a valid response but no shares data
              setShares([]);
            }
          } catch (err) {
            // Handle 404 error (no portfolio yet) gracefully
            if (axios.isAxiosError(err) && err.response?.status === 404) {
              // This means the user doesn't have a portfolio yet - treat as empty
              setShares([]);
            } else {
              // This is a real error we should handle
              throw err;
            }
          }
        } catch (err) {
          console.error('Error fetching portfolio:', err);
          setError(err instanceof Error ? err.message : 'Failed to fetch portfolio data');
          setShares([]); // Empty array instead of simulated data
        } finally {
          setLoading(false);
        }
      };

    const countTotal = (value: number) => {
        setTotalShares((prevTotalShares) => prevTotalShares + value);
    }

    useEffect(() => {
        if (!isLoggedIn) return
        
        fetchData();
      }, [searchTerm, isLoggedIn]);

    // Open buy modal
    const handleBuyClick = () => {
        setSelectedSymbol('');
        setTransactionVolume(0);
        onBuyOpen();
    };

    // Open sell modal with pre-selected symbol
    const handleSellClick = (symbol: string) => {
        setSelectedSymbol(symbol);
        setTransactionVolume(0);
        onSellOpen();
    };

    // Execute buy transaction
    const handleBuy = async () => {
        if (!selectedSymbol || transactionVolume <= 0) {
            toast({
                title: 'Invalid input',
                description: 'Please enter a valid symbol and volume',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        try {
            setLoading(true);
            const apiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || 'https://localhost:55616';
            await axios.patch(`${apiUrl}/portfolio/buy/${selectedSymbol}/${transactionVolume}`);
            
            onBuyClose();
            fetchData();
            
            toast({
                title: 'Purchase successful',
                description: `Bought ${transactionVolume} shares of ${selectedSymbol}`,
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
        } catch (err) {
            console.error('Error buying shares:', err);
            toast({
                title: 'Purchase failed',
                description: err instanceof Error ? err.message : 'An error occurred',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    // Execute sell transaction
    const handleSell = async () => {
        if (!selectedSymbol || transactionVolume <= 0) {
            toast({
                title: 'Invalid input',
                description: 'Please enter a valid volume',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        // Find current shares of this symbol
        const currentShares = shares.find(s => s.symbol === selectedSymbol);
        if (!currentShares || currentShares.volume < transactionVolume) {
            toast({
                title: 'Not enough shares',
                description: `You don't have enough ${selectedSymbol} shares to sell`,
                status: 'warning',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        try {
            setLoading(true);
            const apiUrl = import.meta.env.VITE_API_PORTFOLIO_URL || 'https://localhost:55616';
            await axios.patch(`${apiUrl}/portfolio/sell/${selectedSymbol}/${transactionVolume}`);
            
            onSellClose();
            fetchData();
            
            toast({
                title: 'Sale successful',
                description: `Sold ${transactionVolume} shares of ${selectedSymbol}`,
                status: 'success',
                duration: 3000,
                isClosable: true,
            });
        } catch (err) {
            console.error('Error selling shares:', err);
            toast({
                title: 'Sale failed',
                description: err instanceof Error ? err.message : 'An error occurred',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setLoading(false);
        }
    };

    if (!isLoggedIn) return <div>Please log in to view your portfolio.</div>;
    if (loading) return <div>Loading portfolio data...</div>;
      if (error) return <div>Error: {error}</div>;

    // If no shares, show a helpful empty state
    if (shares.length === 0) {
      return (
        <div className="p-6 bg-white border rounded-lg shadow-sm">
          <h2 className="text-xl font-bold mb-4">Your Portfolio is Empty</h2>
          <p className="mb-4">You don't have any stocks in your portfolio yet. Start adding some to track your investments!</p>
          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <p className="text-blue-700 font-medium">Click "Buy Shares" in the modal to start investing.</p>
          </div>
          <button 
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            onClick={handleBuyClick}
          >
            Buy Your First Stock
          </button>
        </div>
      );
    }

      return(
<Box className="p-4">
  <Flex justifyContent="space-between" alignItems="center" mb={4}>
    <Heading as="h2" size="xl" className="text-xl font-bold mb-2">My Shares</Heading>
    <Button colorScheme="green" onClick={handleBuyClick}>Buy Shares</Button>
  </Flex>

  <table className="min-w-full bg-white border border-gray-300 shadow-md rounded-lg">
    <thead>
      <tr className="bg-gray-100">
        <th className="px-4 py-2">Symbol</th>
        <th className="px-4 py-2">Shares</th>
        <th className="px-4 py-2">Total Price</th>
        <th className="px-4 py-2">Amount</th>
        <th className="px-4 py-2">Actions</th>
      </tr>
    </thead>
    <tbody>
      {shares.map((share, index) => (
        <Share 
          key={index} 
          setSearchTermFinal={setSearchTermFinal} 
          setSearchTerm={setSearchTerm} 
          setCurrentPage={setCurrentPage} 
          getSharePrice={countTotal} 
          share={share} 
          onSellClick={() => handleSellClick(share.symbol)}
        />
      ))}
    </tbody>
  </table>

  <p className="mt-2 text-right font-bold">Total: {totalShares.toFixed(2)}$</p>

  {/* Buy Modal */}
  <Modal isOpen={isBuyOpen} onClose={onBuyClose}>
    <ModalOverlay />
    <ModalContent>
      <ModalHeader>Buy Shares</ModalHeader>
      <ModalCloseButton />
      <ModalBody>
        <FormControl mb={4}>
          <FormLabel>Symbol</FormLabel>
          <Input 
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
            placeholder="e.g. AAPL"
          />
        </FormControl>
        <FormControl>
          <FormLabel>Volume</FormLabel>
          <Input 
            type="number"
            value={transactionVolume || ''}
            onChange={(e) => setTransactionVolume(parseInt(e.target.value) || 0)}
            placeholder="Number of shares"
          />
        </FormControl>
      </ModalBody>
      <ModalFooter>
        <Button colorScheme="blue" mr={3} onClick={handleBuy} isLoading={loading}>
          Buy
        </Button>
        <Button variant="ghost" onClick={onBuyClose}>Cancel</Button>
      </ModalFooter>
    </ModalContent>
  </Modal>

  {/* Sell Modal */}
  <Modal isOpen={isSellOpen} onClose={onSellClose}>
    <ModalOverlay />
    <ModalContent>
      <ModalHeader>Sell {selectedSymbol} Shares</ModalHeader>
      <ModalCloseButton />
      <ModalBody>
        <FormControl>
          <FormLabel>Volume</FormLabel>
          <Input 
            type="number"
            value={transactionVolume || ''}
            onChange={(e) => setTransactionVolume(parseInt(e.target.value) || 0)}
            placeholder="Number of shares"
          />
        </FormControl>
        {selectedSymbol && (
          <Box mt={2}>
            <strong>Available:</strong> {shares.find(s => s.symbol === selectedSymbol)?.volume || 0} shares
          </Box>
        )}
      </ModalBody>
      <ModalFooter>
        <Button colorScheme="red" mr={3} onClick={handleSell} isLoading={loading}>
          Sell
        </Button>
        <Button variant="ghost" onClick={onSellClose}>Cancel</Button>
      </ModalFooter>
    </ModalContent>
  </Modal>
</Box>

      );
}

export default Portfolio;