import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button, TextField } from '@mui/material';
import { AppDispatch } from '../store';
import { ERRORSTATE, createCore, fetchCores, selectCore } from '../store/slices/core';
import { styled } from 'styled-components';

import DatePicker from "react-datepicker";

import "react-datepicker/dist/react-datepicker.css";

function TrxnInput() {
  const [memo, setMemo] = useState<string>("");
  const [startDate, setStartDate] = useState<Date>(new Date());
  const [amount, setAmount] = useState<number>(0);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectCore);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchCores());
    }
  }, [elements, errorState, dispatch]);

  const addAmount = (param1: number) => { // Threshold 0.
    setAmount((am) => (am + param1 >= 0) ? (am + param1) : 0);
  };

  return (
    <TraxionInputDiv>
        <div></div>
        <div>
          <DatePicker selected={startDate} onChange={(date: Date) => setStartDate(date)} />
        </div>
        <TextField label="주기" variant="outlined" value={memo} onChange={(e) => setMemo(e.target.value)}/>
        <TextField label="타입" variant="outlined" value={memo} onChange={(e) => setMemo(e.target.value)}/>
        <div>
          <input type='number' value={amount} onChange={(e) => {
            try{
              setAmount(Number(e.target.value));
            } catch {
              console.log("NaN" + e.target.value);
            };  
          }} pattern="[0-9]+"/>
          <button onClick={() => addAmount(1000)}>+1000</button>
          <button onClick={() => addAmount(-1000)} disabled={amount <= 0}>-1000</button>
          <button onClick={() => addAmount(+10000)}>+10000</button>
          <button onClick={() => addAmount(-10000)} disabled={amount <= 0}>-10000</button>
          <button onClick={() => addAmount(+1000)}>+5000</button>
          <button onClick={() => addAmount(-5000)} disabled={amount <= 0}>-5000</button>
          <button onClick={() => setAmount(0)}>Clear</button>
        </div>
        <TextField label="메모" variant="outlined" value={memo} onChange={(e) => setMemo(e.target.value)}/>
        <Button variant="contained" disabled={memo === ""} onClick={() => {
            dispatch(createCore(memo));
            setMemo("");

            console.log(memo);
            console.log(startDate);
            console.log(amount);
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