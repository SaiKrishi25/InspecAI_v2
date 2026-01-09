import { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  FormControl,
  FormLabel,
  RadioGroup,
  Alert,
  Chip,
  Paper,
  Fade,
} from '@mui/material';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import DownloadIcon from '@mui/icons-material/Download';
import VideocamIcon from '@mui/icons-material/Videocam';
import { captureAndInfer, getCameraStatus } from '../services/api';

const InspecInfer = () => {
  const [cameraAvailable, setCameraAvailable] = useState(false);
  const [loading, setLoading] = useState(false);
  const [detectionMode, setDetectionMode] = useState('fasterrcnn_only');  // Modes: fasterrcnn_only or sam2_only
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    checkCamera();
  }, []);

  const checkCamera = async () => {
    try {
      const status = await getCameraStatus();
      setCameraAvailable(status.status === 'available');
    } catch (err) {
      console.error('Camera check failed:', err);
      setCameraAvailable(false);
    }
  };

  const handleCapture = async () => {
    setLoading(true);
    setError(null);
    setShowResults(false);

    try {
      const response = await captureAndInfer(detectionMode);
      setResults(response);
      setShowResults(true);
    } catch (err: any) {
      console.error('Capture failed:', err);
      setError(err.response?.data?.error || 'Failed to capture and analyze image');
      setShowResults(false);
    } finally {
      setLoading(false);
    }
  };

  const handleNewCapture = () => {
    setShowResults(false);
    setResults(null);
    setError(null);
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Inspec Infer - Real-time Detection
      </Typography>

      <Grid container spacing={3}>
        {/* Left Panel - Camera Feed */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Live Camera Feed
                </Typography>
                <Chip
                  label={cameraAvailable ? 'Camera Ready' : 'Camera Unavailable'}
                  color={cameraAvailable ? 'success' : 'error'}
                  size="small"
                  icon={cameraAvailable ? <CheckCircleIcon /> : <ErrorIcon />}
                />
              </Box>

              <Paper
                elevation={0}
                sx={{
                  border: '2px solid',
                  borderColor: 'primary.main',
                  borderRadius: 2,
                  overflow: 'hidden',
                  backgroundColor: '#000',
                  mb: 2,
                  position: 'relative',
                }}
              >
                <img
                  src={`/video_feed?t=${Date.now()}`}
                  alt="Camera Feed"
                  style={{
                    width: '100%',
                    height: 'auto',
                    display: 'block',
                    minHeight: '300px',
                  }}
                  onError={() => setCameraAvailable(false)}
                  onLoad={() => setCameraAvailable(true)}
                />
                {loading && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      backgroundColor: 'rgba(0,0,0,0.7)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                    }}
                  >
                    <CircularProgress size={60} sx={{ color: 'white', mb: 2 }} />
                    <Typography variant="h6">Processing...</Typography>
                  </Box>
                )}
              </Paper>

              <Box sx={{ mb: 2 }}>
                <FormControl component="fieldset">
                  <FormLabel component="legend" sx={{ fontWeight: 600, mb: 1 }}>
                    Detection Mode
                  </FormLabel>
                  <RadioGroup
                    value={detectionMode}
                    onChange={(e) => setDetectionMode(e.target.value)}
                  >
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          border: '2px solid',
                          borderColor: detectionMode === 'fasterrcnn_only' ? 'primary.main' : '#E0E0E0',
                          borderRadius: 2,
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          '&:hover': {
                            borderColor: 'primary.main',
                            backgroundColor: 'action.hover',
                          },
                        }}
                        onClick={() => setDetectionMode('fasterrcnn_only')}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <input
                            type="radio"
                            value="fasterrcnn_only"
                            checked={detectionMode === 'fasterrcnn_only'}
                            style={{ marginRight: 8 }}
                          />
                          <Box>
                            <Typography variant="body1" sx={{ fontWeight: 600 }}>
                              🔍 Structural Defects (FasterRCNN)
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Detects dents and scratches • Fast (~2-3s)
                            </Typography>
                          </Box>
                        </Box>
                      </Paper>

                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          border: '2px solid',
                          borderColor: detectionMode === 'sam2_only' ? 'primary.main' : '#E0E0E0',
                          borderRadius: 2,
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          '&:hover': {
                            borderColor: 'primary.main',
                            backgroundColor: 'action.hover',
                          },
                        }}
                        onClick={() => setDetectionMode('sam2_only')}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <input
                            type="radio"
                            value="sam2_only"
                            checked={detectionMode === 'sam2_only'}
                            style={{ marginRight: 8 }}
                          />
                          <Box>
                            <Typography variant="body1" sx={{ fontWeight: 600 }}>
                              🎨 Surface Defects (SAM2)
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Detects paint defects, water spots, hazing • Slower (~5-8s)
                            </Typography>
                          </Box>
                        </Box>
                      </Paper>
                    </Box>
                  </RadioGroup>
                </FormControl>
              </Box>

              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <CameraAltIcon />}
                onClick={handleCapture}
                disabled={!cameraAvailable || loading}
                sx={{
                  py: 1.5,
                  fontSize: '1rem',
                  fontWeight: 600,
                }}
              >
                {loading ? 'Analyzing...' : '📸 Capture & Analyze'}
              </Button>
              
              {showResults && (
                <Button
                  variant="outlined"
                  size="large"
                  fullWidth
                  startIcon={<VideocamIcon />}
                  onClick={handleNewCapture}
                  sx={{
                    mt: 1,
                    py: 1.5,
                    fontSize: '1rem',
                    fontWeight: 600,
                  }}
                >
                  🎥 Return to Live Feed
                </Button>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Detection Results
              </Typography>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {loading && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 8 }}>
                  <CircularProgress size={60} sx={{ mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Processing image...
                  </Typography>
                </Box>
              )}

              {!loading && !showResults && !error && (
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    py: 8,
                    border: '2px dashed #E0E0E0',
                    borderRadius: 2,
                  }}
                >
                  <VideocamIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" color="text.primary" gutterBottom>
                    Live Camera Feed Active
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Click "Capture & Analyze" to detect defects
                  </Typography>
                </Box>
              )}

              {showResults && results && !loading && (
                <Fade in={showResults}>
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                        Detection Complete
                      </Typography>
                      <Chip
                        label="Latest Result"
                        color="primary"
                        size="small"
                        icon={<CheckCircleIcon />}
                      />
                    </Box>
                    
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Captured Image
                        </Typography>
                        <Paper
                          elevation={0}
                          sx={{ border: '1px solid #E0E0E0', borderRadius: 1, overflow: 'hidden' }}
                        >
                          <img
                            src={results.original_image}
                            alt="Captured"
                            style={{ width: '100%', display: 'block' }}
                          />
                        </Paper>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Detection Results
                        </Typography>
                        <Paper
                          elevation={0}
                          sx={{ border: '1px solid #E0E0E0', borderRadius: 1, overflow: 'hidden' }}
                        >
                          <img
                            src={results.visualization}
                            alt="Results"
                            style={{ width: '100%', display: 'block' }}
                          />
                        </Paper>
                      </Box>
                    </Grid>
                  </Grid>

                  <Box sx={{ mb: 2 }}>
                    <Alert
                      severity={results.total_defects === 0 ? 'success' : results.total_defects <= 2 ? 'warning' : 'error'}
                      icon={results.total_defects === 0 ? <CheckCircleIcon /> : <ErrorIcon />}
                    >
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        {results.total_defects === 0
                          ? '✅ No Defects Detected - PASS'
                          : results.total_defects <= 2
                          ? `⚠️ ${results.total_defects} Defect(s) Detected - MINOR ISSUES`
                          : `🚨 ${results.total_defects} Defect(s) Detected - FAIL`}
                      </Typography>
                    </Alert>
                  </Box>

                  {results.report_pdf_url && (
                    <Box
                      sx={{
                        p: 2,
                        backgroundColor: 'primary.light',
                        borderRadius: 2,
                        color: 'white',
                      }}
                    >
                      <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                        Inspection Report
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Report ID: <strong>{results.report_id}</strong>
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 2 }}>
                        Total Defects: <strong>{results.total_defects}</strong>
                      </Typography>
                      <Button
                        variant="contained"
                        color="secondary"
                        fullWidth
                        startIcon={<DownloadIcon />}
                        href={results.report_pdf_url}
                        target="_blank"
                        sx={{ backgroundColor: 'white', color: 'primary.main', '&:hover': { backgroundColor: '#f5f5f5' } }}
                      >
                        View Full Report (PDF)
                      </Button>
                    </Box>
                  )}
                  
                    <Box sx={{ mt: 2, p: 2, backgroundColor: '#F5F7FA', borderRadius: 2 }}>
                      <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CheckCircleIcon fontSize="small" color="success" />
                        Camera is ready for next capture. Click "Return to Live Feed" to capture another image.
                      </Typography>
                    </Box>
                  </Box>
                </Fade>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default InspecInfer;

