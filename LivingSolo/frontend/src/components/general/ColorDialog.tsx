import { Button, DialogActions, DialogContent } from "@mui/material";


import { styled } from "styled-components";
import { BootstrapDialog, BootstrapDialogTitle } from "./Dialog";

interface ColorDialogProps {
    open: boolean,
    handleClose: () => void,
    color: string,
    setColor: React.Dispatch<React.SetStateAction<string>>,
  }
  
export const ColorDialog = ({open, handleClose, color, setColor } : ColorDialogProps) => {
  const clearColor = () => {
    setColor('#000000');
  };

  return <div>
    <BootstrapDialog
      onClose={handleClose}
      aria-labelledby="customized-dialog-title"
      open={open}
    >
      <DialogBody>
        <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
          색상 설정
        </BootstrapDialogTitle>
        <DialogContent dividers>
          <SetTagHeaderWrapper>
            <SetTagHeader>
              설정된 색상{'  '}{color}
            </SetTagHeader>
            <ColorClearSpan onClick={() => clearColor()}active={(color !== '#000000').toString()}>Clear</ColorClearSpan>
          </SetTagHeaderWrapper>
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleClose}>
            닫기
          </Button>
        </DialogActions>
      </DialogBody>
    </BootstrapDialog>
  </div>
};

const DialogBody = styled.div`
  width: 500px;
`;
const SetTagHeaderWrapper = styled.div`
  display: flex;
  justify-content: space-between;
`;
const SetTagHeader = styled.span`
`;
const ColorClearSpan = styled.span<{ active: string }>`
    margin-top: 3px;
    font-size: 17px;
    color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    background-color: transparent;
    cursor: ${props => ((props.active === 'true') ? 'pointer' : 'default')};;
    margin-left: 20px;
`;