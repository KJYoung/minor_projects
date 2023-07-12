import React, { useState } from 'react';
import { styled } from 'styled-components';
import { styled as styledMUI } from '@mui/material/styles';
import { RoundButton } from '../../utils/Button';

import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Typography from '@mui/material/Typography';
import { TypeBubbleElement } from '../../store/slices/trxn';
import { TagBubbleCompact } from '../general/TypeBubble';
import { useSelector } from 'react-redux';
import { selectTrxnType } from '../../store/slices/trxnType';

interface TypeDialogProps {
  open: boolean,
  handleClose: () => void,
  initialTags?: TypeBubbleElement[]
}

const TypeDialog = ({open, handleClose, initialTags} : TypeDialogProps) => {
  return <div>
    <BootstrapDialog
      onClose={handleClose}
      aria-labelledby="customized-dialog-title"
      open={open}
    >
      <DialogBody>
        <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
          Tag for Transaction
        </BootstrapDialogTitle>
        <DialogContent dividers>
          <Typography gutterBottom>
            현재 설정된 태그:
            <span>{initialTags?.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
          </Typography>
          <Typography gutterBottom>
            태그 목록.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleClose}>
            Save changes
          </Button>
        </DialogActions>
      </DialogBody>
      
    </BootstrapDialog>
  </div>
}

const BootstrapDialog = styledMUI(Dialog)(({ theme }) => ({
  '& .MuiDialogContent-root': {
    padding: theme.spacing(2),
  },
  '& .MuiDialogActions-root': {
    padding: theme.spacing(1),
  },
}));

const DialogBody = styled.div`
  width: 500px;
`;

export interface DialogTitleProps {
  id: string;
  children?: React.ReactNode;
  onClose: () => void;
}

function BootstrapDialogTitle(props: DialogTitleProps) {
  const { children, onClose, ...other } = props;
  return (
    <DialogTitle sx={{ m: 0, p: 2 }} {...other}>
      {children}
      {onClose ? (
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 8,
            top: 8,
            color: (theme) => theme.palette.grey[500],
          }}
        >
          <CloseIcon />
        </IconButton>
      ) : null}
    </DialogTitle>
  );
}

// Type Input Container.
function TypeInput() {
  const { elements, errorState }  = useSelector(selectTrxnType);

  const [open, setOpen] = React.useState<boolean>(false);
  const [tags, setTags] = useState<TypeBubbleElement[]>([{
    id: 1,
    name: 'Typesss',
    color: '#11df7b',
  }]);

  const handleClickOpen = () => {
    setOpen(true);
  };
  const handleClose = () => {
    setOpen(false);
  };

  return (
    <TypeInputDiv>
        <RoundButton onClick={handleClickOpen}>+</RoundButton>
        {elements.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}
        <TypeDialog open={open} handleClose={handleClose} initialTags={tags} />
    </TypeInputDiv>
  );
}

const TypeInputDiv = styled.div`
    background-color: var(--ls-blue);
    border-radius: 5px;
`;

export default TypeInput;