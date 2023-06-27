import { Button } from "@mui/material";
import { TabState } from "../../App";
import { styled } from "styled-components";

const NavDrawer = ({ toggleDrawer }: {toggleDrawer : any}) => (
    <NavDrawerDiv  role="presentation" onClick={toggleDrawer(false)} onKeyDown={toggleDrawer(false)}>
      <CloseRowDiv>
        <Button onClick={toggleDrawer(false)}>닫기</Button>
      </CloseRowDiv>
      <Button onClick={toggleDrawer(false, TabState.Home)}>Home</Button>
      <Button onClick={toggleDrawer(false, TabState.Transaction)}>Transaction</Button>
      <Button onClick={toggleDrawer(false, TabState.Calendar)}>Calendar</Button>
      <Button onClick={toggleDrawer(false, TabState.Stockpile)}>Stockpile</Button>
    </NavDrawerDiv>
  );

const NavDrawerDiv = styled.div`
    width: 250px;
    display: flex;
    flex-direction: column;
`;

const CloseRowDiv = styled.div`
    width: 100%;
    height: 64px;

    display: flex;
    justify-content: flex-start;
    align-items: center;
    
    background-color: aqua;
`
export default NavDrawer;