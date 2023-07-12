import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { ERRORSTATE } from '../store/slices/core';
import { fetchTrxns, selectTrxn } from '../store/slices/trxn';
import TrxnInput from '../components/Trxn/TrxnInput';
import { TrxnGridHeader, TrxnGridItem, TrxnGridSearcher } from '../components/Trxn/TrxnGrid';
import { fetchTrxnTypes } from '../store/slices/trxnType';

function TrxnMain() {
  const [editID, setEditID] = useState(-1);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectTrxn);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS || errorState === ERRORSTATE.DEFAULT){
      dispatch(fetchTrxns());
      dispatch(fetchTrxnTypes());
    }
  }, [elements, errorState, dispatch]);
  return (
    <div className="App">
      <TrxnInput />
      <TrxnGridSearcher />
      <TrxnGridHeader />
      {elements && elements.map((e, index) => <TrxnGridItem key={e.id} item={e} isEditing={editID === e.id} setEditID={setEditID}/>)}
    </div>
  );
}

export default TrxnMain;