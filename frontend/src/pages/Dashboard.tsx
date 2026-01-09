import { useState, useEffect } from 'react';
import { Box, Grid, Typography, Card, CardContent, CircularProgress } from '@mui/material';
import OverviewCards from '../components/Dashboard/OverviewCards';
import TimeSeriesChart from '../components/Dashboard/TimeSeriesChart';
import DefectDistribution from '../components/Dashboard/DefectDistribution';
import RecentDetections from '../components/Dashboard/RecentDetections';
import { getAnalyticsOverview, getTimeSeriesData, getDefectDistribution, getRecentDetections } from '../services/api';

interface AnalyticsData {
  total_detections: number;
  today_detections: number;
  week_detections: number;
  month_detections: number;
  total_defects: number;
  defects_by_type: Record<string, number>;
  avg_confidence: number;
}

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<AnalyticsData | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [defectDistData, setDefectDistData] = useState([]);
  const [recentDetections, setRecentDetections] = useState([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
    // Refresh data every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [overviewData, timeData, distData, detectionsData] = await Promise.all([
        getAnalyticsOverview(30),
        getTimeSeriesData(30),
        getDefectDistribution(),
        getRecentDetections(5),
      ]);

      setOverview(overviewData);
      setTimeSeriesData(timeData);
      setDefectDistData(distData);
      setRecentDetections(detectionsData);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !overview) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Card sx={{ backgroundColor: 'error.light', color: 'error.contrastText' }}>
          <CardContent>
            <Typography variant="h6">{error}</Typography>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600, color: 'text.primary' }}>
        Analytics Dashboard
      </Typography>

      {overview && (
        <>
          <OverviewCards overview={overview} />

          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12} lg={8}>
              <TimeSeriesChart data={timeSeriesData} />
            </Grid>
            <Grid item xs={12} lg={4}>
              <DefectDistribution data={defectDistData} />
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <RecentDetections detections={recentDetections} />
          </Box>
        </>
      )}
    </Box>
  );
};

export default Dashboard;

