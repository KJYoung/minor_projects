import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { ERRORSTATE } from '../store/slices/core';
import { fetchCombinedTrxns, fetchTrxns, selectTrxn } from '../store/slices/trxn';
import TrxnInput from '../components/Trxn/TrxnInput';
import { TrxnGridHeader, TrxnGridItem, TrxnGridNav} from '../components/Trxn/TrxnGrid';
import { fetchTags, fetchTagsIndex } from '../store/slices/tag';
import { CUR_MONTH } from '../utils/DateTime';

export enum ViewMode {
  Detail, Combined, Graph
};

function TrxnMain() {
  const [editID, setEditID] = useState(-1);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Detail);

  const dispatch = useDispatch<AppDispatch>();
  const { elements, combined, errorState }  = useSelector(selectTrxn);

  useEffect(() => {
    if(errorState === ERRORSTATE.SUCCESS || errorState === ERRORSTATE.DEFAULT){
      dispatch(fetchTrxns({yearMonth: CUR_MONTH}));
      dispatch(fetchCombinedTrxns({yearMonth: CUR_MONTH}));
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
    }
  }, [elements, errorState, dispatch]);
  return (
    <div className="App">
      <TrxnInput />
      <TrxnGridNav viewMode={viewMode} setViewMode={setViewMode}/>
      <TrxnGridHeader viewMode={viewMode}/>
      {viewMode !== ViewMode.Combined && elements && elements.map((e, index) => <TrxnGridItem key={e.id} index={index} item={e} isEditing={editID === e.id} setEditID={setEditID} viewMode={viewMode}/>)}
      {viewMode === ViewMode.Combined && combined.map((e, index) => <div key={index}>{e.toString()}</div>)}
    </div>
  );
}

export default TrxnMain;