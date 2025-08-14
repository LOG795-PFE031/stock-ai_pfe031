import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Text,
  Flex,
  Heading,
  Input,
  Button,
  VStack,
  HStack,
  Avatar,
  IconButton,
  Divider,
  Spinner,
  Tooltip,
  Badge,
  useColorModeValue,
  useToast
} from '@chakra-ui/react';
import { IconSend, IconRobot, IconUser, IconInfoCircle } from '@tabler/icons-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import chatbotService from '../../clients/ChatbotService';
import { AuthServer } from '../../clients/AuthServer';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  isStreaming?: boolean;
}

const Advisor: React.FC = () => {
  console.log("Advisor component is rendering");
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [cursorVisible, setCursorVisible] = useState(true);
  
  const bgColor = useColorModeValue('gray.50', 'gray.800');
  const messageBgUser = useColorModeValue('blue.500', 'blue.400');
  const messageBgAssistant = useColorModeValue('gray.200', 'gray.700');
  
  const toast = useToast();
  
  // Blinking cursor effect
  useEffect(() => {
    if (streamingId) {
      const interval = setInterval(() => {
        setCursorVisible(prev => !prev);
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [streamingId]);
  
  // Initialize the chat with proper user ID from auth system
  useEffect(() => {
    const initializeChat = async () => {
      try {
        setIsLoading(true);
        
        // Check if user is logged in and get username
        const username = AuthServer.getUsername();
        const isAuthenticated = AuthServer.isAuthenticated();
        
        if (isAuthenticated && username) {
          console.log('User is authenticated as:', username);
          
          // Set the user ID in the chat service if needed
          // This should normally happen in the login flow, but we do it here 
          // as a fallback in case the user refreshed the page
          const token = AuthServer.getAuthToken() || '';
          await chatbotService.handleLogin(username, token);
        } else {
          console.log('User is not authenticated, using default profile');
        }
        
        setIsInitialized(true);
      } catch (error) {
        console.error('Error initializing chat:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    initializeChat();
  }, []);
  
  // Add initial welcome message from the assistant
  useEffect(() => {
    if (isInitialized && messages.length === 0) {
      setMessages([
        {
          id: crypto.randomUUID(),
          content: "👋 Hello! I'm your AI Financial Advisor. I can help you with investment advice, portfolio analysis, and answering your financial questions. How can I assist you today?",
          sender: 'assistant',
          timestamp: new Date()
        }
      ]);
    }
  }, [isInitialized, messages.length]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: input,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      // Create a streaming assistant message placeholder
      const assistantMessageId = crypto.randomUUID();
      setStreamingId(assistantMessageId);
      
      const assistantMessage: Message = {
        id: assistantMessageId,
        content: "",
        sender: 'assistant',
        timestamp: new Date(),
        isStreaming: true
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // Use streaming mode with callback
      await chatbotService.sendMessage(input, (token) => {
        setMessages(prev => 
          prev.map(msg => 
            msg.id === assistantMessageId 
              ? { ...msg, content: msg.content + token } 
              : msg
          )
        );
      });
      
      // Mark message as no longer streaming when complete
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessageId 
            ? { ...msg, isStreaming: false } 
            : msg
        )
      );
      
      setStreamingId(null);
    } catch (error) {
      console.error('Error communicating with chatbot:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        content: "I'm sorry, I encountered an error processing your request. Please try again later.",
        sender: 'assistant',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setStreamingId(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Custom rendering components for markdown
  const MarkdownComponents = {
    p: (props: any) => <Text mb={2} {...props} />,
    h1: (props: any) => <Heading as="h1" size="xl" mt={6} mb={4} {...props} />,
    h2: (props: any) => <Heading as="h2" size="lg" mt={5} mb={3} {...props} />,
    h3: (props: any) => <Heading as="h3" size="md" mt={4} mb={2} {...props} />,
    ul: (props: any) => <Box as="ul" pl={4} mb={4} {...props} />,
    ol: (props: any) => <Box as="ol" pl={4} mb={4} {...props} />,
    li: (props: any) => <Box as="li" mb={1} {...props} />,
    a: (props: any) => <Text as="a" color="blue.500" textDecoration="underline" {...props} />,
    strong: (props: any) => <Text as="strong" fontWeight="bold" {...props} />,
    em: (props: any) => <Text as="em" fontStyle="italic" {...props} />,
    code: (props: any) => <Text as="code" bg="gray.100" p={1} borderRadius="sm" {...props} />,
    pre: (props: any) => <Box as="pre" bg="gray.100" p={2} borderRadius="md" overflowX="auto" my={2} {...props} />
  };



  return (
    <Container maxW="container.lg" py={8}>
      
      <Box bg={bgColor} borderRadius="lg" overflow="hidden" boxShadow="md">
        <Box bg="brand.500" color="white" p={4}>
          <Flex align="center">
            <IconRobot size={24} style={{ marginRight: '8px' }} />
            <Heading size="md">Financial Advisor</Heading>
            <Tooltip label="Powered by AI language model with financial expertise" placement="top">
              <IconButton
                aria-label="Information"
                icon={<IconInfoCircle size={20} />}
                variant="ghost"
                color="white"
                ml={2}
                size="sm"
              />
            </Tooltip>
            <Badge ml="auto" colorScheme="green">Online</Badge>
          </Flex>
        </Box>
        
        <Box height="500px" overflowY="auto" p={4}>
          <VStack spacing={4} align="stretch">
            {messages.map(message => (
              <Flex 
                key={message.id} 
                justify={message.sender === 'user' ? 'flex-end' : 'flex-start'}
              >
                <HStack
                  alignItems="flex-start" 
                  maxW="80%" 
                  bg={message.sender === 'user' ? messageBgUser : messageBgAssistant}
                  color={message.sender === 'user' ? 'white' : 'black'}
                  p={3} 
                  borderRadius="lg"
                >
                  {message.sender === 'assistant' && (
                    <Avatar size="sm" icon={<IconRobot size={20} />} bg="brand.500" />
                  )}
                  
                  <Box overflow="hidden">
                    {message.sender === 'assistant' ? (
                      <>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={MarkdownComponents}
                        >
                          {message.content || " "}
                        </ReactMarkdown>
                        {message.isStreaming && cursorVisible && (
                          <Box as="span" display="inline-block" ml={1}>▋</Box>
                        )}
                      </>
                    ) : (
                      <Text>{message.content}</Text>
                    )}
                    <Text fontSize="xs" color={message.sender === 'user' ? 'whiteAlpha.700' : 'gray.500'} textAlign="right">
                      {formatTime(message.timestamp)}
                    </Text>
                  </Box>
                  
                  {message.sender === 'user' && (
                    <Avatar size="sm" icon={<IconUser size={20} />} bg="blue.700" />
                  )}
                </HStack>
              </Flex>
            ))}
            <div ref={messagesEndRef} />
          </VStack>
          
          {isLoading && !streamingId && (
            <Flex justify="flex-start" mt={4}>
              <HStack
                alignItems="center" 
                bg={messageBgAssistant}
                p={3} 
                borderRadius="lg"
              >
                <Avatar size="sm" icon={<IconRobot size={20} />} bg="brand.500" />
                <Spinner size="sm" color="brand.500" />
                <Text>Thinking...</Text>
              </HStack>
            </Flex>
          )}
        </Box>
        
        <Divider />
        
        <Box p={4}>
          <form onSubmit={handleSubmit}>
            <Flex>
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about investments, stock advice, financial planning..."
                mr={2}
                disabled={isLoading || !!streamingId}
                _disabled={{ opacity: 0.7 }}
              />
              <Button
                type="submit"
                colorScheme="blue"
                isDisabled={isLoading || !!streamingId || !input.trim()}
                leftIcon={<IconSend size={16} />}
              >
                Send
              </Button>
            </Flex>
          </form>
          
          <Text fontSize="xs" color="gray.500" mt={2}>
            This advisor provides general information and not personalized investment advice.
            Always consult with a qualified financial professional before making investment decisions.
          </Text>
        </Box>
      </Box>
    </Container>
  );
};

export default Advisor; 
