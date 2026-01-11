import { useState, useRef, useEffect } from 'react';
import {
  Box,
  Drawer,
  IconButton,
  Typography,
  TextField,
  Paper,
  Avatar,
  CircularProgress,
  Chip,
  Divider,
  Alert,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import PersonIcon from '@mui/icons-material/Person';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import DescriptionIcon from '@mui/icons-material/Description';
import { Message, sendMessageToGemini, getSuggestedQuestions } from '../../services/gemini';
import { useLocation, useNavigate } from 'react-router-dom';
import assistantIcon from '../../../assistant_icon.png';

interface DigitalAssistantProps {
  open: boolean;
  onClose: () => void;
}

const ASSISTANT_WIDTH = 400;

const DigitalAssistant = ({ open, onClose }: DigitalAssistantProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your InspecAI Assistant. How can I help you today?',
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const navigate = useNavigate();

  // Extract Report ID from message content
  const extractReportId = (content: string): string | null => {
    // Match Report ID pattern: RPT_YYYYMMDDHHMMSS_XXXXXXXX
    // Also handle escaped underscores (RPT\_...) from markdown formatting
    const reportIdMatch = content.match(/RPT[_\\]*\d{14}[_\\]*[A-Z0-9]{8}/i);
    if (reportIdMatch) {
      // Remove any backslashes from the matched ID
      return reportIdMatch[0].replace(/\\/g, '');
    }
    return null;
  };

  // Handle navigation to report
  const handleViewReport = (reportId: string) => {
    // Open the PDF report directly in a new tab
    window.open(`http://localhost:8000/reports/report_${reportId}.pdf`, '_blank');
  };

  // Handle navigation to reports page
  const handleGoToReports = () => {
    navigate('/reports');
    onClose(); // Close the assistant drawer
  };

  // Get current page for context
  const getCurrentPage = () => {
    const path = location.pathname;
    if (path.includes('dashboard') || path === '/') return 'dashboard';
    if (path.includes('infer')) return 'infer';
    if (path.includes('reports')) return 'reports';
    return 'dashboard';
  };

  const currentPage = getCurrentPage();
  const suggestedQuestions = getSuggestedQuestions(currentPage);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendMessageToGemini([...messages, userMessage], currentPage);
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err.message : 'Failed to get response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputMessage(question);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      variant="persistent"
      sx={{
        width: open ? ASSISTANT_WIDTH : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: ASSISTANT_WIDTH,
          boxSizing: 'border-box',
          marginTop: '64px',
          height: 'calc(100vh - 64px)',
          borderLeft: '1px solid',
          borderColor: 'divider',
          transition: 'transform 0.3s ease-in-out',
        },
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Box
          sx={{
            p: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'primary.main',
            color: 'white',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              component="img"
              src={assistantIcon}
              alt="AI Assistant"
              sx={{
                width: 28,
                height: 28,
                filter: 'brightness(0) invert(1)',
              }}
            />
            <Typography variant="h6" fontWeight={600}>
              AI Assistant
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}>
            <CloseIcon />
          </IconButton>
        </Box>

        {/* Messages Area */}
        <Box
          sx={{
            flexGrow: 1,
            overflowY: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            backgroundColor: 'background.default',
          }}
        >
          {messages.map((message, index) => {
            const reportId = message.role === 'assistant' ? extractReportId(message.content) : null;
            
            return (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  gap: 1,
                  alignItems: 'flex-start',
                  flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                }}
              >
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    backgroundColor: message.role === 'user' ? 'primary.main' : 'secondary.main',
                  }}
                >
                  {message.role === 'user' ? (
                    <PersonIcon fontSize="small" />
                  ) : (
                    <Box
                      component="img"
                      src={assistantIcon}
                      alt="AI"
                      sx={{
                        width: 20,
                        height: 20,
                        filter: 'brightness(0) invert(1)',
                      }}
                    />
                  )}
                </Avatar>
                <Box sx={{ maxWidth: '75%' }}>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 1.5,
                      backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
                      color: message.role === 'user' ? 'white' : 'text.primary',
                    }}
                  >
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {message.content}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        mt: 0.5,
                        display: 'block',
                        opacity: 0.7,
                        fontSize: '0.7rem',
                      }}
                    >
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </Typography>
                  </Paper>
                  
                  {/* Quick action buttons for assistant messages with Report ID */}
                  {reportId && (
                    <Box sx={{ mt: 1.5, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      <Button
                        size="small"
                        variant="contained"
                        color="success"
                        startIcon={<DescriptionIcon />}
                        onClick={() => handleViewReport(reportId)}
                        sx={{ 
                          textTransform: 'none',
                          fontSize: '0.75rem',
                          py: 0.75,
                          px: 2,
                          borderRadius: 2,
                          fontWeight: 600,
                          boxShadow: 2,
                          '&:hover': {
                            boxShadow: 4,
                          }
                        }}
                      >
                         View Report PDF
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="primary"
                        startIcon={<OpenInNewIcon />}
                        onClick={handleGoToReports}
                        sx={{ 
                          textTransform: 'none',
                          fontSize: '0.75rem',
                          py: 0.75,
                          px: 2,
                          borderRadius: 2,
                          fontWeight: 600,
                        }}
                      >
                         Go to Reports Page
                      </Button>
                    </Box>
                  )}
                </Box>
              </Box>
            );
          })}

          {isLoading && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Avatar sx={{ width: 32, height: 32, backgroundColor: 'secondary.main' }}>
                <Box
                  component="img"
                  src={assistantIcon}
                  alt="AI"
                  sx={{
                    width: 20,
                    height: 20,
                    filter: 'brightness(0) invert(1)',
                  }}
                />
              </Avatar>
              <Paper elevation={1} sx={{ p: 1.5 }}>
                <CircularProgress size={20} />
              </Paper>
            </Box>
          )}

          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <div ref={messagesEndRef} />
        </Box>

        {/* Suggested Questions */}
        {messages.length <= 1 && (
          <Box sx={{ px: 2, pb: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              Suggested questions:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {suggestedQuestions.slice(0, 3).map((question, index) => (
                <Chip
                  key={index}
                  label={question}
                  size="small"
                  onClick={() => handleSuggestedQuestion(question)}
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          </Box>
        )}

        <Divider />

        {/* Input Area */}
        <Box sx={{ p: 2, backgroundColor: 'background.paper' }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Ask me anything..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              multiline
              maxRows={3}
              variant="outlined"
            />
            <IconButton
              color="primary"
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              sx={{ alignSelf: 'flex-end' }}
            >
              <SendIcon />
            </IconButton>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
          </Typography>
        </Box>
      </Box>
    </Drawer>
  );
};

export default DigitalAssistant;
export { ASSISTANT_WIDTH };
