import React from 'react';
import { Box, Text, Flex, Link, Stack } from '@chakra-ui/react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <Box as="footer" bg="gray.800" color="gray.200" py={6}>
      <Box maxW="1400px" mx="auto" px={4}>
        <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" align="center">
          <Stack direction="column" mb={{ base: 6, md: 0 }}>
            <Text fontSize="lg" fontWeight="bold">FinancePro</Text>
            <Text fontSize="sm" color="gray.400">Intelligent stock analysis and prediction platform</Text>
          </Stack>
          
          <Stack direction="row" gap={8}>
            <Stack direction="column" gap={2}>
              <Text fontWeight="semibold">Features</Text>
              <Link href="#" fontSize="sm">Stock Prediction</Link>
              <Link href="#" fontSize="sm">Sentiment Analysis</Link>
              <Link href="#" fontSize="sm">Portfolio Management</Link>
            </Stack>
            
            <Stack direction="column" gap={2}>
              <Text fontWeight="semibold">Resources</Text>
              <Link href="#" fontSize="sm">Documentation</Link>
              <Link href="#" fontSize="sm">API</Link>
              <Link href="#" fontSize="sm">Help Center</Link>
            </Stack>
            
            <Stack direction="column" gap={2}>
              <Text fontWeight="semibold">Company</Text>
              <Link href="#" fontSize="sm">About Us</Link>
              <Link href="#" fontSize="sm">Privacy Policy</Link>
              <Link href="#" fontSize="sm">Terms of Service</Link>
            </Stack>
          </Stack>
        </Flex>
        
        <Box height="1px" bg="gray.700" my={6} />
        
        <Text textAlign="center" fontSize="sm" color="gray.500">
          © {currentYear} FinancePro. All rights reserved. Data provided for educational purposes only.
        </Text>
      </Box>
    </Box>
  );
};

export default Footer; 