import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { ERRORSTATE, coreActions } from './store/slices/core';
import { RootState } from '.';

function App() {
  const [name, setName] = useState("");
  const [editName, setEditName] = useState("");
  const [editID, setEditID] = useState(-1);

  const dispatch = useDispatch();
  const { coreElements, coreState } = useSelector(({ core }: RootState) => ({ coreElements: core.elements, coreState: core.errorState }));

  useEffect(() => {
    if(coreState === ERRORSTATE.SUCCESS){
      dispatch(coreActions.getElements());
    }
  }, [coreState, dispatch]);
  return (
    <div className="App">
      <button onClick={() => { dispatch(coreActions.getElements()); }}>FETCH!</button>
      <input value={name} onChange={(e) => {setName(e.target.value)}}/><button disabled={name===""} onClick={() => { dispatch(coreActions.createCore({name})); setName(""); }}>SEND</button>
      {coreElements && coreElements.map((e, index) => 
        <div>
          {editID === e.id ? 
            <>
              <input value={editName} onChange={(e) => setEditName(e.target.value)}/>
              <button disabled={editName === ""} onClick={() => {
                dispatch(coreActions.editCore({id: e.id, name: editName}));
                setEditID(-1); setEditName("");
              }}>수정 완료</button>
            </>
            : 
            <span>{e.id} | {e.name}</span>
          }
            {editID !== e.id && <button onClick={async () => { setEditID(e.id); setEditName(e.name)}}>수정</button>}
            <button onClick={async () => dispatch(coreActions.deleteCore({id: e.id})) }>삭제</button>
        </div>
      )}
    </div>
  );
}

export default App;
