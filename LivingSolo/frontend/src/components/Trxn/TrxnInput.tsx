import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button, TextField } from '@mui/material';
import { AppDispatch } from '../../store';
import { ERRORSTATE } from '../../store/slices/core';
import { createTrxn, fetchTrxns, selectTrxn } from '../../store/slices/trxn';
import { styled } from 'styled-components';

import DatePicker from "react-datepicker";

import "react-datepicker/dist/react-datepicker.css";
import { GetDateTimeFormat2Django } from '../../utils/DateTime';
import TypeInput from './TypeInput';
import { NewAmountInput } from './AmountInput';
import { TypeBubbleElement } from '../../store/slices/trxnType';

function TrxnInput() {
  const [memo, setMemo] = useState<string>("");
  const [trxnDate, setTrxnDate] = useState<Date>(new Date());
  const [realDate, setRealDate] = useState<Date>(new Date());
  const [amount, setAmount] = useState<number>(0);
  const [tags, setTags] = useState<TypeBubbleElement[]>([]);
  const [isPeriodic, setIsPeriodic] = useState<boolean>(false); // is periodic transaction?
  const [hasDiffTime, setHasDiffTime] = useState<boolean>(false); // 값을 지출한 날과 그 값을 실제로 소비한 날이 다른가?
  const [period, setPeriod] = useState<number>(0);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectTrxn);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchTrxns());
    }
  }, [elements, errorState, dispatch]);

  return (
    <TrxnInputDiv>
      <Trxn1stRowDiv>
        <div>
          <DatePicker selected={trxnDate} onChange={(date: Date) => setTrxnDate(date)} />
          <label htmlFor="hasDiffTimeChecker">명목/실질 구분?</label>
          <input type="checkbox" id="hasDiffTimeChecker" checked={hasDiffTime} onChange={(e) => setHasDiffTime((hdt) => !hdt)} />
          {hasDiffTime && <DatePicker selected={realDate} onChange={(date: Date) => setRealDate(date)} />}
        </div>
        <div>
          <label htmlFor="isPeriodic">주기성?</label>
          <input type="checkbox" id="isPeriodic" checked={isPeriodic} onChange={(e) => setIsPeriodic((ip) => !ip)} />
          <input placeholder='주기' type='number' value={period.toString()} onChange={(e) => {
              try{
                const num = Number(e.target.value);
                setPeriod(num >= 0 ? num : 0);
              } catch {
                console.log("NaN" + e.target.value);
              };  
            }} pattern="[0-9]+" min={0} disabled={!isPeriodic}/>
        </div>
        <TypeInput tags={tags} setTags={setTags}/>
        <NewAmountInput amount={amount} setAmount={setAmount}/>
      </Trxn1stRowDiv>
      <Trxn2ndRowDiv>
        <TextField label="메모" variant="outlined" value={memo} onChange={(e) => setMemo(e.target.value)}/>
        <Button variant="contained" disabled={memo === ""} onClick={() => {
            dispatch(createTrxn({
              memo,
              amount,
              period,
              date: GetDateTimeFormat2Django(trxnDate),
              type: tags
            }));
            setMemo(""); setAmount(0); setPeriod(0); setTags([]);
        }}>기입</Button>
      </Trxn2ndRowDiv>
    </TrxnInputDiv>
  );
}

const TrxnInputDiv = styled.div`
  display: flex;
  flex-direction: column;

  padding-top: 10px;
  padding-bottom: 10px;
  border-bottom: 1px solid gray;

  > div {
    padding-left: 5%;
    padding-right: 5%;
  }
  margin-bottom: 20px;
`;

const Trxn1stRowDiv = styled.div`
  display: grid;
  grid-template-columns: 2fr 2fr 2fr 2fr;
  grid-column-gap: 30px;
  justify-content: space-around;
  width: 100%;

  margin-top: 5px;
  margin-bottom: 5px;
`;
const Trxn2ndRowDiv = styled.div`
  display: grid;
  grid-template-columns: 12fr 2fr;
  grid-column-gap: 30px;
  justify-content: space-around;
  width: 100%;

  margin-top: 5px;
  margin-bottom: 5px;
`;

export default TrxnInput;
