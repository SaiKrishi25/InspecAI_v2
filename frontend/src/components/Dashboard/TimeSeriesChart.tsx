import { Card, CardContent, Typography, Box } from '@mui/material';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format, parseISO } from 'date-fns';

interface TimeSeriesData {
  date: string;
  detection_count: number;
  defect_count: number;
}

interface TimeSeriesChartProps {
  data: TimeSeriesData[];
}

const TimeSeriesChart = ({ data }: TimeSeriesChartProps) => {
  // Format data for chart
  const chartData = data.map(item => ({
    ...item,
    date: item.date ? format(parseISO(item.date), 'MMM dd') : '',
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ pb: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' } }}>
          Detection Trends (Last 30 Days)
        </Typography>
        <Box sx={{ width: '100%', height: { xs: 280, sm: 320, md: 380 } }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart 
              data={chartData} 
              margin={{ top: 10, right: 20, left: 10, bottom: 5 }}
            >
              <defs>
                <linearGradient id="colorDetections" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2196F3" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#2196F3" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="colorDefects" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#F44336" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#F44336" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" />
              <XAxis
                dataKey="date"
                stroke="#757575"
                tick={{ fontSize: 11 }}
              />
              <YAxis 
                stroke="#757575" 
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid #E0E0E0',
                  borderRadius: 8,
                }}
              />
              <Legend wrapperStyle={{ fontSize: '14px' }} />
              <Area
                type="monotone"
                dataKey="detection_count"
                stroke="#2196F3"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorDetections)"
                name="Detections"
              />
              <Area
                type="monotone"
                dataKey="defect_count"
                stroke="#F44336"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorDefects)"
                name="Defects"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default TimeSeriesChart;

