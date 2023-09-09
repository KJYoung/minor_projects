import { Button, DialogActions, DialogContent } from "@mui/material";
import { ChromePicker } from 'react-color';

import { styled } from "styled-components";
import { BootstrapDialog, BootstrapDialogTitle } from "./Dialog";
import { IPropsActive } from "../../utils/Interfaces";

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
          <SetColorHeaderWrapper>
            <SetColorHeader>
              설정된 색상{'  '}{color}
            </SetColorHeader>
            <ColorClearSpan onClick={() => clearColor()} isActive={(color !== '#000000').toString()}>Clear</ColorClearSpan>
          </SetColorHeaderWrapper>
          <ColorPickerWrapper>
            <ChromePicker color={color} onChange={color => setColor(color.hex)} />
          </ColorPickerWrapper>
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

const SetColorHeaderWrapper = styled.div`
  display: flex;
  justify-content: space-between;
`;

const SetColorHeader = styled.span`
`;

const ColorClearSpan = styled.span<IPropsActive>`
    margin-top: 3px;
    font-size: 17px;
    background-color: transparent;
    margin-left: 20px;

    color: ${props => ((props.isActive === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
    cursor: ${props => ((props.isActive === 'true') ? 'pointer' : 'default')};;
`;

const ColorPickerWrapper = styled.div`
  width: 100%;
  height: 100%;
  margin-top: 40px;

  display: flex;
  justify-content: center;
  align-items: center;
`