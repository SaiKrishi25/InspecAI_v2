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
          <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid #E0E0E0' }}>
            <Table>
              <TableHead sx={{ backgroundColor: '#F5F7FA' }}>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Timestamp</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Detection ID</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Image ID</TableCell>
                  <TableCell sx={{ fontWeight: 600 }} align="center">Defects</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Mode</TableCell>
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
                    <TableCell>
                      {detection.timestamp
                        ? format(parseISO(detection.timestamp), 'MMM dd, yyyy HH:mm')
                        : 'N/A'}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                      {detection.detection_id || 'N/A'}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                      {detection.image_id || 'N/A'}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={detection.total_defects}
                        size="small"
                        color={detection.total_defects === 0 ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={detection.detection_mode === 'fasterrcnn_only' ? 'Fast' : 'Full'}
                        size="small"
                        variant="outlined"
                        color="primary"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={getStatusLabel(detection.total_defects)}
                        size="small"
                        color={getStatusColor(detection.total_defects)}
                        sx={{ fontWeight: 600, minWidth: 60 }}
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

