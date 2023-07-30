import React, { useState } from 'react';
import TrxnMain from './containers/TrxnMain';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Drawer from '@mui/material/Drawer';
import { styled } from 'styled-components';
import NavDrawer from './components/general/NavDrawer';
import TodoMain from './containers/TodoMain';

export enum TabState { DEPRECATED, Home, Transaction, Calendar, Stockpile, Community };
const tab2Str = (e: TabState) => {
  switch (e) {
    case TabState.Home:
      return 'Home';
    case TabState.Transaction:
      return 'Transaction';
    case TabState.Calendar:
      return 'Calendar(ToDo)';
    case TabState.Stockpile:
      return 'Stockpile';
    case TabState.Community:
      return 'Community';
    default:
      return 'Unknown';
  }
}

function App() {
  const [tabState, setTabState] = useState<TabState>(TabState.Calendar);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  const toggleDrawer = (open: boolean, newTabState?: TabState) =>
    (event: React.KeyboardEvent | React.MouseEvent) => {
      if(event.type === 'keydown' && ((event as React.KeyboardEvent).key === 'Tab' || (event as React.KeyboardEvent).key === 'Shift')) return;
      if(newTabState) setTabState(newTabState); // Set New Tab State.
      setIsDrawerOpen(open);
    };

  return (
      <AppDiv>
        {/* Material Style Nav Bar */}
        <Box>
          <AppBar position="static" sx={{ height: 64 }}>
            <Toolbar>
              <IconButton
                size="large"
                edge="start"
                color="inherit"
                aria-label="menu"
                sx={{ mr: 2 }}
                onClick={toggleDrawer(true)}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                {tab2Str(tabState)}
              </Typography>
              <Button color="inherit">Login</Button>
            </Toolbar>
          </AppBar>
        </Box>
        {/* Main Contents */}
        {tabState === TabState.Home && <span>Home</span>}
        {tabState === TabState.Transaction && <TrxnMain />}
        {tabState === TabState.Calendar && <TodoMain />}
        {tabState === TabState.Stockpile && <span>Stockpile</span>}
        {tabState === TabState.Community && <span>Community</span>}
        
        <Drawer open={isDrawerOpen}>
          <NavDrawer toggleDrawer={toggleDrawer}/>
        </Drawer>
      </AppDiv>
      );
    }

const AppDiv = styled.div`
  width: 100%;
  height: 100%;
`;
export default App;
