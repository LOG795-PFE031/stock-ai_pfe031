import { Box } from '@chakra-ui/react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

type DataPoint = { date: string; price: number };

type Props = {
  data: DataPoint[];
  predictedData?: DataPoint[];
  height?: string | number;
};

const PriceAreaChart = ({ data, predictedData = [], height = 400 }: Props) => {
  const mergedData = mergeDataWithPredictions(data, predictedData);

  return (
    <Box height={height} mb={6}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={mergedData}
          margin={{
            top: 10,
            right: 30,
            left: 0,
            bottom: 0,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          {/* Actual price */}
          <Area
            type="monotone"
            dataKey="price"
            stroke="#0066ff"
            fill="#0066ff"
            fillOpacity={0.2}
            name="Actual Price"
          />
          {/* Predicted price */}
          {predictedData.length > 0 && (
            <Area
              type="monotone"
              dataKey="predictedPrice"
              stroke="#FFA500"
              strokeWidth={2}
              fill="none"
              name="Predicted Price"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PriceAreaChart;

// Merge historical and predicted data by date
function mergeDataWithPredictions(
  actual: DataPoint[],
  predicted: DataPoint[]
): Array<{ date: string; price?: number; predictedPrice?: number }> {
  const map = new Map<string, { date: string; price?: number; predictedPrice?: number }>();

  actual.forEach(({ date, price }) => {
    map.set(date, { date, price: Math.round(price * 100) / 100 });
  });

  predicted.forEach(({ date, price }) => {
    const existing = map.get(date) ?? { date };
    existing.predictedPrice = Math.round(price * 100) / 100;
    map.set(date, existing);
  });

  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
}
