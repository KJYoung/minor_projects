import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ERRORSTATE, createCore, deleteCore, editCore, fetchCores, selectCore } from './store/slices/core';
import { AppDispatch } from './store';
import { Button } from '@mui/material';

function App() {
  const [name, setName] = useState("");
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
      <Button variant={"contained"} onClick={() => { dispatch(fetchCores()); }}>FETCH!</Button>
      <input value={name} onChange={(e) => {setName(e.target.value)}}/>
      <Button variant={"contained"} disabled={name===""} onClick={() => { dispatch(createCore(name)); setName(""); }}>SEND</Button>
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

export default App;
