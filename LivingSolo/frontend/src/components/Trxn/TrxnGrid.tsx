import React, { useState } from 'react';
import { styled } from 'styled-components';
import { Button } from '@mui/material';
import { useDispatch } from 'react-redux';
import { TrxnElement, deleteTrxn, editTrxn, fetchTrxns } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TypeBubble';
import { GetDateTimeFormatFromDjango } from '../../utils/DateTime';
import { EditAmountInput } from './AmountInput';

interface TrxnGridItemProps {
    item: TrxnElement,
    isEditing: boolean,
    setEditID: React.Dispatch<React.SetStateAction<number>>
};

export function TrxnGridSearcher() {
    const dispatch = useDispatch<AppDispatch>();
    return <>
        <Button variant={"contained"} onClick={() => { dispatch(fetchTrxns()); }}>FETCH!</Button>
    </>
};

export function TrxnGridHeader() {
  return (
    <TrxnGridHeaderDiv>
        <span>ID</span>
        <span>Date</span>
        <span>Period</span>
        <span>Type</span>
        <span>Amount</span>
        <span>Memo</span>
        <span>...</span>
    </TrxnGridHeaderDiv>
  );
};
export function TrxnGridItem({ item, isEditing, setEditID }: TrxnGridItemProps) {
  const [trxnItem, setTrxnItem] = useState<TrxnElement>(item);

  const dispatch = useDispatch<AppDispatch>();
  return (
    <TrxnGridItemDiv key={item.id}>
        <span>{item.id}</span>
        <span>{GetDateTimeFormatFromDjango(item.date, true)}</span>
        <span>{item.period > 0 ? item.period : '-'}</span>
        <span>{item.type.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
        
        {/* AMOUNT */}
        {isEditing ? <div>
            <EditAmountInput amount={trxnItem.amount} setAmount={setTrxnItem} />

              {/* <input value={trxnItem.amount} onChange={(e) => setTrxnItem((item) => { return { ...item, amount: Number(e.target.value) } })}/> */}
            </div> 
            : 
            <span>{item.amount}</span>
        }

        {/* MEMO */}
        {isEditing ? <div>
              <input value={trxnItem.memo} onChange={(e) => setTrxnItem((item) => { return { ...item, memo: e.target.value } })}/>
            </div> 
            : 
            <span>{item.memo}</span>
        }
        <div>
            {isEditing && <Button variant={"contained"} disabled={trxnItem.memo === ""} onClick={() => {
                    dispatch(editTrxn(trxnItem));
                    setEditID(-1);
                }}>수정 완료</Button>}
            {!isEditing && <Button onClick={async () => { setEditID(item.id); setTrxnItem(item); }} variant={"contained"}>수정</Button>}
            {!isEditing && <Button onClick={async () => {
                if (window.confirm('정말 기록을 삭제하시겠습니까?')) {
                    dispatch(deleteTrxn(item.id));
                }}} variant={"contained"}>삭제</Button>}
        </div>
    </TrxnGridItemDiv>
  );
};

const TrxnGridTemplate = styled.div`
    display: grid;
    grid-template-columns: 1fr 3fr 2fr 2fr 2fr 5fr 2fr;
    padding-left: 70px;
    padding-right: 70px;   
`;

const TrxnGridHeaderDiv = styled(TrxnGridTemplate)`
    > span {
        border: 1px solid red;
        text-align: center;
        font-size: 22px;
    }
`;
const TrxnGridItemDiv = styled(TrxnGridTemplate)`
    > span {
        border: 1px solid red;
        text-align: center;
        font-size: 22px;
    }
    input {
        width: 100%;
        height: 100%;
    }
`;

