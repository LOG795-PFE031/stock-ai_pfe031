import React, { useState, useEffect } from 'react';
import { Link as RouterLink, useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Flex,
  Stack,
  Button,
  Input,
  Link as ChakraLink,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Avatar,
  Text
} from '@chakra-ui/react';
import { IconSearch, IconChevronDown, IconUser, IconLogout, IconSettings } from '@tabler/icons-react';
import { AuthServer } from '../../clients/AuthServer';

// Add custom event for auth state changes
const authStateChangeEvent = new Event('authStateChange');

const Header = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [searchedTicker, setSearchedTicker] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  // Function to check authentication status
  const checkAuthStatus = () => {
    if (AuthServer.isAuthenticated()) {
      setIsLoggedIn(true);
      const savedUsername = AuthServer.getUsername();
      if (savedUsername) {
        setUsername(savedUsername);
      }
    } else {
      setIsLoggedIn(false);
      setUsername('');
    }
  };

  // Check auth status on component mount and route changes
  useEffect(() => {
    checkAuthStatus();
    
    // Add event listener for auth state changes
    window.addEventListener('authStateChange', checkAuthStatus);
    
    // Clean up event listener on unmount
    return () => {
      window.removeEventListener('authStateChange', checkAuthStatus);
    };
  }, [location.pathname]); // Re-check when route changes

  const handleLogout = () => {
    AuthServer.logout();
    setIsLoggedIn(false);
    setUsername('');
    
    // Dispatch event to notify other components
    window.dispatchEvent(authStateChangeEvent);
    
    navigate('/');
  };

  return (
    <Box as="header" bg="white" boxShadow="sm" position="sticky" top={0} zIndex={10}>
      <Flex 
        maxW="7xl" 
        mx="auto" 
        px={4} 
        py={3} 
        align="center" 
        justify="space-between"
      >
        <Box fontWeight="bold" fontSize="xl">
          <ChakraLink as={RouterLink} to="/" _hover={{ textDecoration: 'none' }}>
            Stock Companion
          </ChakraLink>
        </Box>

        <Flex align="center" gap={8}>
          <Stack direction="row" gap={6} display={{ base: 'none', md: 'flex' }}>
            <ChakraLink as={RouterLink} to="/markets" fontWeight="medium">
              Markets
            </ChakraLink>
            
            <Menu>
              <MenuButton as={ChakraLink} fontWeight="medium">
                Portfolio <IconChevronDown size={14} style={{ display: 'inline' }} />
              </MenuButton>
              <MenuList>
                <MenuItem as={RouterLink} to="/portfolio">
                  Overview
                </MenuItem>
                <MenuItem as={RouterLink} to="/portfolio/performance">
                  Performance Analytics
                </MenuItem>
              </MenuList>
            </Menu>
            
            <ChakraLink as={RouterLink} to="/advisor" fontWeight="medium">
              AI Advisor
            </ChakraLink>
            <ChakraLink as={RouterLink} to="/news" fontWeight="medium">
              News
            </ChakraLink>
          </Stack>

          <Flex gap={2} align="center">
            <Flex position="relative" display={{ base: 'none', md: 'flex' }}>
              <Input 
                placeholder="Search..." 
                size="sm" 
                width="auto" 
                borderRadius="md"
                value={searchedTicker}
                onChange={(e)=>setSearchedTicker(e.target.value)}
              />
              <IconButton
                icon={<IconSearch size={18} />}
                size="sm"
                aria-label="Search"
                position="absolute"
                right={1}
                top={1}
                zIndex={2}
                variant="ghost"
                onClick={()=>navigate(`/stock/${searchedTicker.toUpperCase()}`)}
              />
            </Flex>

            {isLoggedIn ? (
              <Menu>
                <MenuButton
                  as={Button}
                  size="sm"
                  variant="outline"
                  colorScheme="brand"
                  rightIcon={<IconChevronDown size={14} />}
                >
                  <Flex align="center" gap={2}>
                    <Avatar size="xs" name={username} bg="brand.500" />
                    <Text fontSize="sm" display={{ base: 'none', md: 'block' }}>{username}</Text>
                  </Flex>
                </MenuButton>
                <MenuList>
                  <MenuItem icon={<IconUser size={18} />}>
                    Profile
                  </MenuItem>
                  <MenuItem icon={<IconSettings size={18} />}>
                    Settings
                  </MenuItem>
                  <MenuItem icon={<IconLogout size={18} />} onClick={handleLogout}>
                    Logout
                  </MenuItem>
                </MenuList>
              </Menu>
            ) : (
              <Button 
                size="sm" 
                colorScheme="brand"
                as={RouterLink}
                to="/login"
              >
                Login
              </Button>
            )}
          </Flex>
        </Flex>
      </Flex>
    </Box>
  );
};

export default Header; 