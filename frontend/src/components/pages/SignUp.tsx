import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Heading,
  Text,
  useToast,
  Container,
  InputGroup,
  InputRightElement,
  IconButton,
  Flex,
  Link,
} from '@chakra-ui/react';
import { IconEye, IconEyeOff } from '@tabler/icons-react';
import { AuthServer } from '../../clients/AuthServer';

const SignUp: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();
  
  // Initialize the AuthServer with the API base URL
  const authServer = new AuthServer(import.meta.env.VITE_API_AUTH_URL || 'https://localhost:55604');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!username || !password) {
      toast({
        title: 'Error',
        description: 'Please enter both username and password',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    setIsLoading(true);
    
    try {
      console.log('Attempting registry with server:', import.meta.env.VITE_API_AUTH_URL || 'https://localhost:55604');
      
      await authServer.createUser(username, password);
      
      toast({
        title: 'Sign Up Success',
        description: 'Congratulation, your account is now set. You are redirected to the login pages.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      
      // Notify other components about auth state change
      window.dispatchEvent(new Event('authStateChange'));
      
      // Redirect to login
      navigate('/login');
    } catch (error) {
      console.error('All Sign Up attempts failed:', error);
      
      let errorMessage = 'An unknown error occurred';
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      toast({
        title: 'Sign Up Failed',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const togglePasswordVisibility = () => setShowPassword(!showPassword);

  return (
    <Container maxW="md" py={12}>
      <VStack spacing={8} align="stretch">
        <Box textAlign="center">
          <Heading size="xl">Welcome On Board!</Heading>
          <Text mt={2} color="gray.600">Create your credentials to have a full access</Text>
        </Box>
        
        <Box as="form" onSubmit={handleSubmit} p={8} borderWidth={1} borderRadius="lg" boxShadow="md">
          <VStack spacing={4}>
            <FormControl id="username" isRequired>
              <FormLabel>Username</FormLabel>
              <Input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
              />
            </FormControl>
            
            <FormControl id="password" isRequired>
              <FormLabel>Password</FormLabel>
              <InputGroup>
                <Input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                />
                <InputRightElement>
                  <IconButton
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    icon={showPassword ? <IconEyeOff size={18} /> : <IconEye size={18} />}
                    variant="ghost"
                    onClick={togglePasswordVisibility}
                    size="sm"
                  />
                </InputRightElement>
              </InputGroup>
            </FormControl>
            
            <Button
              type="submit"
              colorScheme="brand"
              width="full"
              mt={4}
              isLoading={isLoading}
              loadingText="Signing in"
            >
              Sign Up
            </Button>
          </VStack>
        </Box>
        
        <Flex justify="center">
          <Text>Already have an account? </Text>
          <Link color="brand.500" ml={1} href="/login">
            Login
          </Link>
        </Flex>

      </VStack>
    </Container>
  );
};

export default SignUp;
