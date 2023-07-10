import React, { useState } from 'react';
import { styled } from 'styled-components';
import { Button } from '@mui/material';
import { useDispatch } from 'react-redux';
import { TrxnElement, deleteTrxn, editTrxn, fetchTrxns } from '../../store/slices/trxn';
import { AppDispatch } from '../../store';
import { TagBubbleCompact } from '../general/TypeBubble';

interface TrxnGridItemProps {
    item: TrxnElement,
    isEditing: boolean,
    setEditID: React.Dispatch<React.SetStateAction<number>>
};

export function TrxnGridHeader() {
  const dispatch = useDispatch<AppDispatch>();
  return (
    <TrxnGridHeaderDiv>
        <Button variant={"contained"} onClick={() => { dispatch(fetchTrxns()); }}>FETCH!</Button>
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
  const [editMemo, setEditMemo] = useState("");

  const dispatch = useDispatch<AppDispatch>();
  return (
    <TrxnGridItemDiv key={item.id}>
        <div></div>
        <span>{item.id}</span>
        <span>{item.date}</span>
        <span>{item.period > 0 ? item.period : '-'}</span>
        <span>{item.type.map((ee) => <TagBubbleCompact key={ee.id} color={ee.color}>{ee.name}</TagBubbleCompact>)}</span>
        <span>{item.amount}</span>
        <span>{item.memo}</span>
        <div key={item.id}>
          {isEditing &&
            <>
              <input value={editMemo} onChange={(e) => setEditMemo(e.target.value)}/>
              <Button variant={"contained"} disabled={editMemo === ""} onClick={() => {
                    // dispatch(editTrxn({id: item.id, memo: editMemo}));
                    setEditID(-1); setEditMemo("");
                }}>수정 완료</Button>
            </>
          }
            {!isEditing && <Button onClick={async () => { setEditID(item.id); setEditMemo(item.memo)}} variant={"contained"}>수정</Button>}
            {!isEditing && <Button onClick={async () => dispatch(deleteTrxn(item.id))} variant={"contained"}>삭제</Button>}
        </div>
    </TrxnGridItemDiv>
  );
};

const TrxnGridTemplate = styled.div`
    display: grid;
    grid-template-columns: 1fr 1fr 3fr 2fr 2fr 2fr 5fr 2fr;
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
`;

