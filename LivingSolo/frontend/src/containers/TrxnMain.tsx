import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button } from '@mui/material';
import { AppDispatch } from '../store';
import { ERRORSTATE, deleteCore, editCore, fetchCores, selectCore } from '../store/slices/core';
import TrxnInput from '../components/TrxnInput';

function TrxnMain() {
  const [editName, setEditName] = useState("");
  const [editID, setEditID] = useState(-1);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectCore);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS){
      dispatch(fetchCores());
    }
  }, [elements, errorState, dispatch]);
  return (
    <div className="App">
      <TrxnInput />
      <Button variant={"contained"} onClick={() => { dispatch(fetchCores()); }}>FETCH!</Button>
      {elements && 
        elements.map((e, index) => 
        <div key={index}>
          {editID === e.id ? 
            <>
              <input value={editName} onChange={(e) => setEditName(e.target.value)}/>
              <Button variant={"contained"} disabled={editName === ""} onClick={() => {
                dispatch(editCore({id: e.id, name: editName}));
                setEditID(-1); setEditName("");
              }}>수정 완료</Button>
            </>
            : 
            <span>{e.id} | {e.name}</span>
          }
            {editID !== e.id && <Button onClick={async () => { setEditID(e.id); setEditName(e.name)}} variant={"contained"}>수정</Button>}
            <Button onClick={async () => dispatch(deleteCore(e.id))} variant={"contained"}>삭제</Button>
        </div>
      )
      }
    </div>
  );
}

export default TrxnMain;
