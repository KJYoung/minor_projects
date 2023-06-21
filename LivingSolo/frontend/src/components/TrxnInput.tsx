import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button, TextField } from '@mui/material';
import { AppDispatch } from '../store';
import { ERRORSTATE, createCore, fetchCores, selectCore } from '../store/slices/core';
import { styled } from 'styled-components';

function TrxnInput() {
  const [memo, setMemo] = useState("");

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectCore);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchCores());
    }
  }, [elements, errorState, dispatch]);
  return (
    <TraxionInputDiv>
        <div></div>
        <TextField label="날짜" variant="outlined"/>
        <TextField label="주기" variant="outlined" />
        <TextField label="타입" variant="outlined" />
        <TextField label="금액" variant="outlined" />
        <TextField label="메모" variant="outlined" value={memo} onChange={(e) => setMemo(e.target.value)}/>
        <Button variant="contained" disabled={memo === ""} onClick={() => {
            dispatch(createCore(memo));
            setMemo("");
        }}>기입</Button>
    </TraxionInputDiv>
  );
}

const TraxionInputDiv = styled.div`
    display: grid;
    grid-template-columns: 2fr 2fr 2fr 2fr 4fr 12fr 2fr;
    grid-column-gap: 30px;
    justify-content: space-around;
    width: 100%;

    margin-top: 5px;
    margin-bottom: 15px;
`;

export default TrxnInput;
