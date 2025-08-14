import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Button } from '@chakra-ui/react';

type Share = {
    symbol: string;
    volume: number;
};

type params = {
    setSearchTermFinal: React.Dispatch<React.SetStateAction<string>>;
    setSearchTerm: React.Dispatch<React.SetStateAction<string>>;
    setCurrentPage: React.Dispatch<React.SetStateAction<string>>;
    getSharePrice: (value: number) => void;
    share: Share;
    onSellClick?: () => void;
};

const Share: React.FC<params> = ({ 
    setSearchTermFinal, 
    setSearchTerm, 
    setCurrentPage, 
    getSharePrice, 
    share, 
    onSellClick 
}) => {
    const [stockPrice, setStockPrice] = useState<number>(0);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const apiUrl = import.meta.env.VITE_API_STOCKS_URL || 'https://localhost:55611';
                const live = new Date().toISOString();
                const response = await axios.get(`${apiUrl}/stocks/${share.symbol}/${live}`);
                setStockPrice(response.data.value);
                getSharePrice(share.volume * response.data.value);
            } catch (err) {
                console.error('Failed to fetch stock data', err);
                setStockPrice(0);
                getSharePrice(0);
            }
        };
        fetchData();
    }, [share, getSharePrice]);

    return (
        <tr className="border-b hover:bg-gray-50">
            <td className="px-4 py-2 text-blue-600 font-bold hover:underline cursor-pointer" 
                onClick={() => {
                    setSearchTermFinal(share.symbol);
                    setSearchTerm(share.symbol);
                    setCurrentPage("market");
                }}>
                {share.symbol}
            </td>
            <td className="px-4 py-2">{share.volume}</td>
            <td className="px-4 py-2">${(share.volume * stockPrice).toFixed(2)}</td>
            <td className="px-4 py-2">${stockPrice.toFixed(2)}</td>
            <td className="px-4 py-2">
                <Button 
                    colorScheme="red" 
                    size="sm" 
                    onClick={onSellClick}
                >
                    Sell
                </Button>
            </td>
        </tr>
    );
};

export default Share;

