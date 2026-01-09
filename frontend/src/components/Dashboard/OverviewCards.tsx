import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';

interface OverviewCardsProps {
  overview: {
    total_detections: number;
    today_detections: number;
    week_detections: number;
    month_detections: number;
    total_defects: number;
    avg_confidence: number;
  };
}

const OverviewCards = ({ overview }: OverviewCardsProps) => {
  const cards = [
    {
      title: 'Total Detections',
      value: overview.total_detections,
      subtitle: `${overview.today_detections} today`,
      icon: <TrendingUpIcon sx={{ fontSize: 40 }} />,
      color: '#2196F3',
      bgColor: '#E3F2FD',
    },
    {
      title: 'This Week',
      value: overview.week_detections,
      subtitle: 'detections',
      icon: <CalendarTodayIcon sx={{ fontSize: 40 }} />,
      color: '#4CAF50',
      bgColor: '#E8F5E9',
    },
    {
      title: 'This Month',
      value: overview.month_detections,
      subtitle: 'detections',
      icon: <CalendarTodayIcon sx={{ fontSize: 40 }} />,
      color: '#FF9800',
      bgColor: '#FFF3E0',
    },
    {
      title: 'Total Defects',
      value: overview.total_defects,
      subtitle: `${(overview.avg_confidence * 100).toFixed(1)}% avg confidence`,
      icon: <ErrorOutlineIcon sx={{ fontSize: 40 }} />,
      color: '#F44336',
      bgColor: '#FFEBEE',
    },
  ];

  return (
    <Grid container spacing={3}>
      {cards.map((card, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card
            sx={{
              height: '100%',
              transition: 'transform 0.2s, box-shadow 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 16px rgba(0,0,0,0.15)',
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    backgroundColor: card.bgColor,
                    color: card.color,
                    borderRadius: 2,
                    p: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {card.icon}
                </Box>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: card.color, mb: 0.5 }}>
                {card.value.toLocaleString()}
              </Typography>
              <Typography variant="h6" sx={{ color: 'text.primary', mb: 0.5 }}>
                {card.title}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                {card.subtitle}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

export default OverviewCards;

