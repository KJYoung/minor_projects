import { Button, Typography } from "@mui/material";
import { TabState } from "../../App";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { styled } from "styled-components";
import { faClose } from "@fortawesome/free-solid-svg-icons";

const NavDrawer = ({ toggleDrawer }: {toggleDrawer : any}) => (
    <NavDrawerDiv  role="presentation" onClick={toggleDrawer(false)} onKeyDown={toggleDrawer(false)}>
      <CloseRowDiv onClick={toggleDrawer(false)}>
        <div>
            <FontAwesomeIcon icon={faClose} />
            <Typography variant="h6" component="div">Close</Typography>
        </div>
      </CloseRowDiv>
      <Button onClick={toggleDrawer(false, TabState.Home)}>Home</Button>
      <Button onClick={toggleDrawer(false, TabState.Tag)}>Tag</Button>
      <Button onClick={toggleDrawer(false, TabState.Transaction)}>Transaction</Button>
      <Button onClick={toggleDrawer(false, TabState.Calendar)}>Calendar</Button>
      <Button onClick={toggleDrawer(false, TabState.Stockpile)}>Stockpile</Button>
      <Button onClick={toggleDrawer(false, TabState.Community)}>Community</Button>
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

    background-color: var(--ls-blue);
    color: white;

    cursor: pointer;
    >div{
        width: 100%;
        height: 40px;

        display: flex;
        align-items: center;
        >svg{
            padding-bottom: 2px;
            margin-left: 20px;
            margin-right: 16px;
            width: 30px;
            height: 30px;
        }
    }
`
export default NavDrawer;