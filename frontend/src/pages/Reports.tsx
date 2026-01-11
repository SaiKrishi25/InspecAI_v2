import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  CircularProgress,
  Button,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getAllReports, searchReports } from '../services/api';
import { format, parseISO } from 'date-fns';

interface Report {
  id: number;
  report_id: string;
  detection_id: string;
  pdf_path: string;
  timestamp: string;
  total_defects: number;
  image_id: string;
  created_at: string;
}

const Reports = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getAllReports(100, 0);
      setReports(data);
    } catch (err) {
      console.error('Error fetching reports:', err);
      setError('Failed to load reports');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      fetchReports();
      return;
    }

    try {
      setLoading(true);
      const data = await searchReports(searchTerm);
      setReports(data);
    } catch (err) {
      console.error('Search failed:', err);
      setError('Search failed');
    } finally {
      setLoading(false);
    }
  };

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

  const handleDownloadReport = (reportId: string) => {
    window.open(`/reports/report_${reportId}.pdf`, '_blank');
  };

  return (
    <Box sx={{ width: '100%', maxWidth: '100%' }}>
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' },
        justifyContent: 'space-between', 
        alignItems: { xs: 'flex-start', sm: 'center' }, 
        mb: { xs: 2, sm: 3 },
        gap: { xs: 2, sm: 0 }
      }}>
        <Typography 
          variant="h4" 
          sx={{ 
            fontWeight: 600,
            fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
          }}
        >
          Inspection Reports
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={fetchReports}
          disabled={loading}
          fullWidth={false}
          sx={{ minWidth: { xs: '100%', sm: 'auto' } }}
        >
          Refresh
        </Button>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            fullWidth
            placeholder="Search by Report ID or Image ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
              endAdornment: searchTerm && (
                <InputAdornment position="end">
                  <Button onClick={handleSearch} variant="contained" size="small">
                    Search
                  </Button>
                </InputAdornment>
              ),
            }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Box sx={{ textAlign: 'center', py: 8, color: 'error.main' }}>
              <Typography>{error}</Typography>
            </Box>
          ) : reports.length === 0 ? (
            <Box
              sx={{
                textAlign: 'center',
                py: 8,
                color: 'text.secondary',
                border: '2px dashed #E0E0E0',
                borderRadius: 2,
              }}
            >
              <Typography variant="h6">No reports found</Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                {searchTerm ? 'Try a different search term' : 'Perform some detections to generate reports'}
              </Typography>
            </Box>
          ) : (
            <TableContainer sx={{ overflowX: 'auto', maxWidth: '100%' }}>
              <Table sx={{ minWidth: { xs: 300, sm: 650 } }}>
                <TableHead sx={{ backgroundColor: '#F5F7FA' }}>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Generated</TableCell>
                    <TableCell sx={{ fontWeight: 600, display: { xs: 'none', sm: 'table-cell' }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Report ID</TableCell>
                    <TableCell sx={{ fontWeight: 600, display: { xs: 'none', md: 'table-cell' }, fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>Image ID</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }} align="center">
                      Defects
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }} align="center">
                      Status
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: { xs: '0.75rem', sm: '0.875rem' } }} align="center">
                      Actions
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {reports.map((report) => (
                    <TableRow
                      key={report.id}
                      sx={{
                        '&:hover': { backgroundColor: '#F5F7FA' },
                      }}
                    >
                      <TableCell sx={{ fontSize: { xs: '0.7rem', sm: '0.875rem' } }}>
                        {report.created_at
                          ? format(parseISO(report.created_at), 'MMM dd, yyyy HH:mm')
                          : report.timestamp
                          ? format(parseISO(report.timestamp), 'MMM dd, yyyy HH:mm')
                          : 'N/A'}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: { xs: '0.7rem', sm: '0.875rem' }, fontWeight: 600, display: { xs: 'none', sm: 'table-cell' } }}>
                        {report.report_id}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: { xs: '0.7rem', sm: '0.875rem' }, display: { xs: 'none', md: 'table-cell' } }}>
                        {report.image_id || 'N/A'}
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={report.total_defects || 0}
                          size="small"
                          color={report.total_defects === 0 ? 'success' : 'error'}
                          sx={{ fontWeight: 600, minWidth: 40 }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={getStatusLabel(report.total_defects || 0)}
                          size="small"
                          color={getStatusColor(report.total_defects || 0)}
                          sx={{ fontWeight: 600, minWidth: 60 }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleDownloadReport(report.report_id)}
                            title="View Report"
                          >
                            <VisibilityIcon />
                          </IconButton>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleDownloadReport(report.report_id)}
                            title="Download Report"
                          >
                            <DownloadIcon />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {reports.length > 0 && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Showing {reports.length} report{reports.length !== 1 ? 's' : ''}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default Reports;

