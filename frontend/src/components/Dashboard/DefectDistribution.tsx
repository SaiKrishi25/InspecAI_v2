import { Card, CardContent, Typography, Box } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface DefectData {
  defect_type: string;
  count: number;
  avg_confidence: number;
}

interface DefectDistributionProps {
  data: DefectData[];
}

const COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4'];

const DefectDistribution = ({ data }: DefectDistributionProps) => {
  // Format data for chart
  const chartData = data.map(item => ({
    name: item.defect_type || 'Unknown',
    value: item.count,
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ pb: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, fontSize: { xs: '1rem', sm: '1.25rem' } }}>
          Defect Type Distribution
        </Typography>
        <Box sx={{ width: '100%', height: { xs: 280, sm: 320, md: 380 } }}>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="42%"
                  labelLine={false}
                  label={false}
                  outerRadius={85}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E0E0E0',
                    borderRadius: 8,
                    padding: '8px 12px',
                  }}
                  formatter={(value: number) => [`${value} defects`, '']}
                />
                <Legend
                  verticalAlign="bottom"
                  height={60}
                  iconType="circle"
                  wrapperStyle={{ 
                    fontSize: '13px',
                    paddingTop: '10px',
                  }}
                  formatter={(value: string, entry: any) => `${value}: ${entry.payload.value}`}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: 'text.secondary',
              }}
            >
              <Typography>No defect data available</Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default DefectDistribution;

