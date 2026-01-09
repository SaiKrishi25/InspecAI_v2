import { createTheme } from '@mui/material/styles';

// InspecAI Blue and White Theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196F3',      // Blue
      light: '#64B5F6',     // Light Blue
      dark: '#1976D2',      // Dark Blue
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#64B5F6',      // Light Blue
      light: '#90CAF9',
      dark: '#42A5F5',
      contrastText: '#ffffff',
    },
    background: {
      default: '#F5F7FA',   // Light grey background
      paper: '#FFFFFF',     // White paper
    },
    text: {
      primary: '#212121',   // Dark grey text
      secondary: '#757575', // Medium grey text
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    info: {
      main: '#2196F3',
    },
    success: {
      main: '#4caf50',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      color: '#212121',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      color: '#212121',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: '#212121',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#212121',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: '#212121',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      color: '#212121',
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#263238', // Dark grey-blue header
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#FFFFFF',
          borderRight: '1px solid #E0E0E0',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 6,
          fontWeight: 600,
        },
        contained: {
          boxShadow: '0 2px 4px rgba(33,150,243,0.3)',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(33,150,243,0.4)',
          },
        },
      },
    },
  },
});

export default theme;

