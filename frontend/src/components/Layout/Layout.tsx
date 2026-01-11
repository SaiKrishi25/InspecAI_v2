import { useState } from 'react';
import { Box, useMediaQuery, useTheme } from '@mui/material';
import Header from './Header';
import Sidebar from './Sidebar';
import DigitalAssistant, { ASSISTANT_WIDTH } from '../DigitalAssistant/DigitalAssistant';

interface LayoutProps {
  children: React.ReactNode;
}

const DRAWER_WIDTH = 240;

const Layout = ({ children }: LayoutProps) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleAssistantToggle = () => {
    setAssistantOpen(!assistantOpen);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: 'background.default' }}>
      <Header onMenuClick={handleDrawerToggle} onAssistantClick={handleAssistantToggle} />
      <Sidebar 
        mobileOpen={mobileOpen} 
        onDrawerToggle={handleDrawerToggle}
        isMobile={isMobile}
      />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: { xs: 2, sm: 3, md: 3 }, // Top padding
          pr: assistantOpen ? 0 : { xs: 2, sm: 3, md: 3 }, // Remove right padding when assistant is open
          pb: { xs: 2, sm: 3, md: 3 }, // Bottom padding
          pl: { xs: 2, sm: 2, md: 2 }, // Minimal left padding (16px)
          marginTop: '64px', // Height of AppBar
          marginLeft: 0, // No left margin - Drawer handles positioning
          minHeight: 'calc(100vh - 64px)',
          width: assistantOpen ? `calc(100% - ${ASSISTANT_WIDTH}px)` : '100%', // Adjust width when assistant is open
          maxWidth: '100%',
          overflowX: 'hidden',
          boxSizing: 'border-box',
          transition: 'width 0.3s ease-in-out, padding-right 0.3s ease-in-out', // Smooth transition
        }}
      >
        {children}
      </Box>
      <DigitalAssistant open={assistantOpen} onClose={handleAssistantToggle} />
    </Box>
  );
};

export default Layout;

