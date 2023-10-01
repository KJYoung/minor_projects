import { Button, Typography } from "@mui/material";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { styled } from "styled-components";
import { faClose } from "@fortawesome/free-solid-svg-icons";
import { useNavigate } from "react-router-dom";

const NavDrawer = ({ toggleDrawer }: {toggleDrawer : any}) => {
  const navigate = useNavigate();

  return <NavDrawerDiv  role="presentation" onClick={toggleDrawer(false)} onKeyDown={toggleDrawer(false)}>
    <CloseRowDiv onClick={toggleDrawer(false)}>
      <div>
          <FontAwesomeIcon icon={faClose} />
          <Typography variant="h6" component="div">Close</Typography>
      </div>
    </CloseRowDiv>
    <Button onClick={() => navigate('/home')}>Home</Button>
    <Button onClick={() => navigate('/tag')}>태그(Tag)</Button>
    <Button onClick={() => navigate('/trxn')}>가계부(Transaction)</Button>
    <Button onClick={() => navigate('/todo')}>투두(Calendar)</Button>
    <Button onClick={() => navigate('/stockpile')}>Stockpile</Button>
    <Button onClick={() => navigate('/community')}>Community</Button>
  </NavDrawerDiv>
};

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