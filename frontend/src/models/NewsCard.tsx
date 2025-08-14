import React from 'react';
import {
  Box,
  Text,
  Flex,
  Badge,
  Card,
  CardBody,
  CardFooter,
  Stack,
  Heading,
  Avatar,
  HStack,
  Icon,
  Link,
  Tooltip
} from "@chakra-ui/react";
import { ExternalLinkIcon, InfoIcon, TimeIcon, LinkIcon } from '@chakra-ui/icons';

type NewsData = {
    title: string;
    publishedAt: string;
    opinion: number;
    url?: string;
    source?: string;
    confidence?: number;
};

interface NewsCardProps {
    data: NewsData;
}

const NewsCard: React.FC<NewsCardProps> = ({ data }) => {
    const formattedData = {
        publishedAt: new Date(data.publishedAt).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }),
        title: data.title,
        opinion: data.opinion,
        url: data.url || '#',
        source: data.source || 'Unknown Source',
        confidence: data.confidence || 0
    };

    const getSentimentColor = (opinion: number): string => {
        if (opinion > 0.3) return "green";
        if (opinion < -0.3) return "red";
        return "gray";
    };

    const getSentimentLabel = (opinion: number): string => {
        if (opinion > 0.3) return "Positive";
        if (opinion < -0.3) return "Negative";
        return "Neutral";
    };

    const sentimentColor = getSentimentColor(formattedData.opinion);
    const sentimentLabel = getSentimentLabel(formattedData.opinion);

    const handleCardClick = () => {
        if (formattedData.url && formattedData.url !== '#') {
            window.open(formattedData.url, '_blank', 'noopener,noreferrer');
        }
    };

    const getSourceInitial = () => {
        const source = formattedData.source;
        if (source.includes('Yahoo')) return 'Y';
        if (source.includes('Bloomberg')) return 'B';
        if (source.includes('Reuters')) return 'R';
        if (source.includes('CNBC')) return 'C';
        if (source.includes('Fool')) return 'F';
        if (source.includes('WSJ')) return 'W';
        return source.charAt(0).toUpperCase();
    };

    return (
        <Card 
            size="sm" 
            variant="outline" 
            borderRadius="md" 
            boxShadow="sm" 
            cursor="pointer" 
            onClick={handleCardClick}
            h="full"
            position="relative"
            _hover={{ 
                boxShadow: "md",
            }}
        >
            <Box position="absolute" top="2" right="2" zIndex="1">
                <HStack spacing="1">
                    <Avatar 
                        size="xs" 
                        name={formattedData.source} 
                        bg="gray.700" 
                        color="white" 
                        fontSize="xs"
                    >
                        {getSourceInitial()}
                    </Avatar>
                    <Badge 
                        size="sm" 
                        bg="white" 
                        color="gray.700" 
                        fontSize="xs" 
                        borderRadius="full" 
                        boxShadow="sm"
                        border="1px" 
                        borderColor="gray.100"
                        px={1.5}
                        py={0.5}
                        maxW="80px"
                        isTruncated
                    >
                        {formattedData.source}
                    </Badge>
                </HStack>
            </Box>
            
            <CardBody pt={8}>
                <Stack spacing={2}>
                    <Flex gap={1} flexWrap="wrap">
                        <Badge 
                            colorScheme={sentimentColor} 
                            variant="subtle"
                            display="flex"
                            alignItems="center"
                            px={1.5}
                            py={0.5}
                            borderRadius="full"
                            fontSize="2xs"
                        >
                            {sentimentLabel}
                        </Badge>
                        
                        {formattedData.confidence !== undefined && (
                            <Tooltip label={`Confidence score: ${Math.round(formattedData.confidence * 100)}%`} placement="top">
                                <Flex alignItems="center">
                                    <Icon as={InfoIcon} color="gray.400" boxSize={3} mr={1} />
                                    <Text fontSize="xs" color="gray.600">
                                        {Math.round(formattedData.confidence * 100)}%
                                    </Text>
                                </Flex>
                            </Tooltip>
                        )}
                    </Flex>
                    
                    <Heading
                        as="h3"
                        fontSize="sm"
                        noOfLines={3}
                        color="gray.800"
                        _hover={{ color: "brand.500" }}
                        transition="color 0.2s"
                    >
                        {formattedData.url && formattedData.url !== '#' ? (
                            <Link href={formattedData.url} isExternal color="gray.800" _hover={{ color: "brand.500", textDecoration: "none" }}>
                                {formattedData.title}
                                <Icon as={ExternalLinkIcon} mx="2px" verticalAlign="text-bottom" boxSize={3} />
                            </Link>
                        ) : (
                            formattedData.title
                        )}
                    </Heading>
                </Stack>
                
                <Flex 
                    mt={3} 
                    pt={2} 
                    justifyContent="flex-end" 
                    alignItems="center"
                    borderTop="1px" 
                    borderColor="gray.100"
                >
                    <HStack spacing={1} fontSize="xs" color="gray.500">
                        <Icon as={TimeIcon} boxSize="3" />
                        <Text>{formattedData.publishedAt}</Text>
                    </HStack>
                </Flex>
            </CardBody>
            
            {formattedData.url && formattedData.url !== '#' && (
                <CardFooter 
                    pt={0} 
                    pb={2} 
                    px={4}
                    mt={-2}
                    borderTop="none"
                >
                    <Flex justifyContent="space-between" alignItems="center" width="full">
                        <Text 
                            color="brand.500" 
                            fontSize="xs" 
                            fontWeight="medium"
                            _hover={{ textDecoration: "underline" }}
                        >
                            Read article
                        </Text>
                        <Icon as={ExternalLinkIcon} color="brand.500" boxSize="3" />
                    </Flex>
                </CardFooter>
            )}
        </Card>
    );
};

export default NewsCard;