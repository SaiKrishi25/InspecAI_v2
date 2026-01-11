import { Card, CardContent, Typography, Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import { format, parseISO } from 'date-fns';

interface Detection {
  id: number;
  detection_id: string;
  timestamp: string;
  image_id: string;
  total_defects: number;
  detection_mode: string;
}

interface RecentDetectionsProps {
  detections: Detection[];
}

const RecentDetections = ({ detections }: RecentDetectionsProps) => {
  const getStatusColor = (defectCount: number) => {
    if (defectCount === 0) return 'success';
    if (defectCount <= 2) return 'warning';
    return 'error';
  };

  const getStatusLabel = (defectCount: number) => {
    if (defectCount === 0) return 'PASS';
    if (defectCount <= 2) return 'MINOR';
    return 'FAIL';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Recent Detections
        </Typography>
        {detections.length > 0 ? (
          <TableContainer 
            component={Paper} 
            elevation={0} 
            sx={{ 
              border: '1px solid #E0E0E0',
              overflowX: 'auto',
              maxWidth: '100%'
            }}
          >
            <Table sx={{ minWidth: { xs: 300, sm: 650 } }}>
              <TableHead sx={{ backgroundColor: '#F5F7FA' }}>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Timestamp</TableCell>
                  <TableCell sx={{ fontWeight: 600, display: { xs: 'none', sm: 'table-cell' } }}>Detection ID</TableCell>
                  <TableCell sx={{ fontWeight: 600, display: { xs: 'none', md: 'table-cell' } }}>Image ID</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">Defects</TableCell>
                  <TableCell sx={{ fontWeight: 600, display: { xs: 'none', sm: 'table-cell' } }}>Mode</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {detections.map((detection) => (
                  <TableRow
                    key={detection.id}
                    sx={{
                      '&:hover': { backgroundColor: '#F5F7FA' },
                      cursor: 'pointer',
                    }}
                  >
                    <TableCell sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
                      {detection.timestamp
                        ? format(parseISO(detection.timestamp), 'MMM dd, yyyy HH:mm')
                        : 'N/A'}
                    </TableCell>
                    <TableCell 
                      sx={{ 
                        fontFamily: 'monospace', 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        display: { xs: 'none', sm: 'table-cell' }
                      }}
                    >
                      {detection.detection_id || 'N/A'}
                    </TableCell>
                    <TableCell 
                      sx={{ 
                        fontFamily: 'monospace', 
                        fontSize: { xs: '0.75rem', sm: '0.875rem' },
                        display: { xs: 'none', md: 'table-cell' }
                      }}
                    >
                      {detection.image_id || 'N/A'}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={detection.total_defects}
                        size="small"
                        color={detection.total_defects === 0 ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>
                      <Chip
                        label={detection.detection_mode === 'fasterrcnn_only' ? 'RCNN' : 'SAM2'}
                        size="small"
                        variant="outlined"
                        color={detection.detection_mode === 'fasterrcnn_only' ? 'primary' : 'secondary'}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={getStatusLabel(detection.total_defects)}
                        size="small"
                        color={getStatusColor(detection.total_defects)}
                        sx={{ fontWeight: 600, minWidth: { xs: 50, sm: 60 } }}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Box
            sx={{
              p: 4,
              textAlign: 'center',
              color: 'text.secondary',
              border: '1px dashed #E0E0E0',
              borderRadius: 2,
            }}
          >
            <Typography>No recent detections available</Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentDetections;

