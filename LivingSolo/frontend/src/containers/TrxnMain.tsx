import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { ERRORSTATE } from '../store/slices/core';
import { fetchTrxns, selectTrxn } from '../store/slices/trxn';
import TrxnInput from '../components/Trxn/TrxnInput';
import { TrxnGridHeader, TrxnGridItem, TrxnGridNav} from '../components/Trxn/TrxnGrid';
import { fetchTags, fetchTagsIndex } from '../store/slices/tag';
import { CUR_MONTH } from '../utils/DateTime';

export enum ViewMode {
  Detail, Graph
};

function TrxnMain() {
  const [editID, setEditID] = useState(-1);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Detail);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, errorState }  = useSelector(selectTrxn);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS || errorState === ERRORSTATE.DEFAULT){
      dispatch(fetchTrxns({yearMonth: CUR_MONTH}));
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
    }
  }, [elements, errorState, dispatch]);
  return (
    <div className="App">
      <TrxnInput />
      <TrxnGridNav viewMode={viewMode} setViewMode={setViewMode}/>
      <TrxnGridHeader viewMode={viewMode}/>
      {elements && elements.map((e, index) => <TrxnGridItem key={e.id} index={index} item={e} isEditing={editID === e.id} setEditID={setEditID} viewMode={viewMode}/>)}
    </div>
  );
}

export default TrxnMain;