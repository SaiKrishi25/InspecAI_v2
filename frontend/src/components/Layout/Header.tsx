import { AppBar, Toolbar, Typography, Box, IconButton, useMediaQuery, useTheme, Badge, Tooltip } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import SettingsIcon from '@mui/icons-material/Settings';
import assistantIcon from '../../../assistant_icon.png';

interface HeaderProps {
  onMenuClick: () => void;
  onAssistantClick: () => void;
}

const Header = ({ onMenuClick, onAssistantClick }: HeaderProps) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        {isMobile && (
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={onMenuClick}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
        )}
        
        <Typography
          variant="h5"
          component="div"
          sx={{
            fontWeight: 700,
            letterSpacing: '0.5px',
            background: '#FFFFFF',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: { xs: '1.1rem', sm: '1.25rem', md: '1.5rem' },
          }}
        >
          INSPEC AI
        </Typography>
        
        <Box sx={{ flexGrow: 1 }} />
        
        <Box sx={{ display: 'flex', gap: { xs: 0.5, sm: 1 } }}>
          {!isMobile && (
            <IconButton color="inherit" aria-label="search">
              <SearchIcon />
            </IconButton>
          )}
          <Tooltip title="AI Assistant" arrow>
            <IconButton 
              color="inherit" 
              aria-label="digital assistant"
              onClick={onAssistantClick}
              sx={{
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              <Badge color="secondary" variant="dot">
                <Box
                  component="img"
                  src={assistantIcon}
                  alt="AI Assistant"
                  sx={{
                    width: 24,
                    height: 24,
                    filter: 'brightness(0) invert(1)', // Makes it white to match header icons
                  }}
                />
              </Badge>
            </IconButton>
          </Tooltip>
          <IconButton color="inherit" aria-label="settings">
            <SettingsIcon />
          </IconButton>
          <IconButton color="inherit" aria-label="user account">
            <AccountCircleIcon />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

