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
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import { ReactNotifications } from 'react-notifications-component';
import TagMain from './containers/TagMain';

function App() {
  return <>
    {/* <GlobalStyles /> */}
    <BrowserRouter>
      <ReactNotifications />
      <Routes>
        <Route
          path="*"
          element={
            <>
              <WebComponent />
            </>
          }
        />
      </Routes>
    </BrowserRouter>
  </>
};

const WebComponent = () => {
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const location = useLocation();

  const toggleDrawer = (open: boolean) =>
    (event: React.KeyboardEvent | React.MouseEvent) => {
      if(event.type === 'keydown' && ((event as React.KeyboardEvent).key === 'Tab' || (event as React.KeyboardEvent).key === 'Shift')) return;
      setIsDrawerOpen(open);
    };
  
  return <Routes>
    <Route
      path="*"
      element={
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
                {location.pathname.substring(1)}
              </Typography>
              <Button color="inherit">Login</Button>
            </Toolbar>
          </AppBar>
        </Box>
        {/* Main Contents */}
        <Routes>
            <Route path="community" element={<span>Community</span>} />
            <Route path="stockpile" element={<span>Stockpile</span>} />
            <Route path="todo" element={<TodoMain />} />
            <Route path="trxn" element={<TrxnMain />} />
            <Route path="tag" element={<TagMain />} />
            <Route path="*" element={<span>Home</span>} />
        </Routes>
        
        <Drawer open={isDrawerOpen}>
          <NavDrawer toggleDrawer={toggleDrawer}/>
        </Drawer>
        
        <Footer></Footer>
      </AppDiv>
      }
    />
  </Routes>
};

const AppDiv = styled.div`
  width: 100%;
  height: 100%;
`;

const Footer = styled.div`
  background-color: gray;
  width: 100%;
  height: 50px;
  margin-top: 40px;
`;
export default App;
