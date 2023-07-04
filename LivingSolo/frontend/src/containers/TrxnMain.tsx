import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button } from '@mui/material';
import { AppDispatch } from '../store';
import { ERRORSTATE, deleteTrxn, fetchTrxns, selectTrxn } from '../store/slices/trxn';
import TrxnInput from '../components/TrxnInput';
import { TagBubbleCompact } from '../components/general/TypeBubble';

function TrxnMain() {
  const [editName, setEditName] = useState("");
  const [editID, setEditID] = useState(-1);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectTrxn);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchTrxns());
    }
  }, [elements, errorState, dispatch]);
  return (
    <div className="App">
      <TrxnInput />
      <Button variant={"contained"} onClick={() => { dispatch(fetchTrxns()); }}>FETCH!</Button>
      <span>ID | DATE | TYPE | MEMO</span>
      {elements && 
        elements.map((e, index) => 
        <div key={index}>
          {editID === e.id ? 
            <>
              <input value={editName} onChange={(e) => setEditName(e.target.value)}/>
              <Button variant={"contained"} disabled={editName === ""} onClick={() => {
                // dispatch(editTrxn({id: e.id, memo: editName}));
                setEditID(-1); setEditName("");
              }}>수정 완료</Button>
            </>
            : 
            <span>{e.id} | {e.date} | {e.memo} | {e.type.map((ee) => {
              return <TagBubbleCompact color={ee.color}>{ee.name}</TagBubbleCompact>
            })}</span>
          }
            {/* {editID !== e.id && <Button onClick={async () => { setEditID(e.id); setEditName(e.name)}} variant={"contained"}>수정</Button>} */}
            <Button onClick={async () => dispatch(deleteTrxn(e.id))} variant={"contained"}>삭제</Button>
        </div>
      )
      }
    </div>
  );
}

export default TrxnMain;
