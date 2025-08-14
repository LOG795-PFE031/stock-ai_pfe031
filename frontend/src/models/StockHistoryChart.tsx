import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush,
  ResponsiveContainer
} from 'recharts';

type StockDataPoint = {
  date: string; // or Date if you'd like to parse into Date objects
  close: number;
  volume: number;
  // include open, high, low, if you prefer
};

interface StockHistoryChartProps {
  data: StockDataPoint[];
}

const StockHistoryChart: React.FC<StockHistoryChartProps> = ({ data }) => {
  return (
    <div style={{ width: '100%', height: 400 }}>
      {/* ResponsiveContainer adjusts to the parent container’s size */}
      <ResponsiveContainer>
        <LineChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <Legend verticalAlign="top" />
          
          {/*
            You can have multiple Lines or just one.
            Let's show the close price.
          */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#8884d8"
            activeDot={{ r: 8 }}
            dot={false}
          />
          
          {/*
            Brush allows the user to zoom in/out by selecting
            a portion of the data range below the chart.
          */}
          <Brush dataKey="date" height={30} stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StockHistoryChart;