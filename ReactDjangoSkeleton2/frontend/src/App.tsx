import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ERRORSTATE, createCore, deleteCore, editCore, fetchCores, selectCore } from './store/slices/core';
import { AppDispatch } from './store';

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
      <button onClick={() => { dispatch(fetchCores()); }}>FETCH!</button>
      <input value={name} onChange={(e) => {setName(e.target.value)}}/><button disabled={name===""} onClick={() => { dispatch(createCore(name)); setName(""); }}>SEND</button>
      {elements && 
        elements.map((e, index) => 
        <div key={index}>
          {editID === e.id ? 
            <>
              <input value={editName} onChange={(e) => setEditName(e.target.value)}/>
              <button disabled={editName === ""} onClick={() => {
                dispatch(editCore({id: e.id, name: editName}));
                setEditID(-1); setEditName("");
              }}>수정 완료</button>
            </>
            : 
            <span>{e.id} | {e.name}</span>
          }
            {editID !== e.id && <button onClick={async () => { setEditID(e.id); setEditName(e.name)}}>수정</button>}
            <button onClick={async () => dispatch(deleteCore(e.id)) }>삭제</button>
        </div>
      )
      }
    </div>
  );
}

export default App;
