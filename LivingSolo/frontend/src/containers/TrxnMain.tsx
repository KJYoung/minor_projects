import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchCombinedTrxns, fetchTrxns, selectTrxn } from '../store/slices/trxn';
import TrxnInput from '../components/Trxn/TrxnInput';
import { fetchTagPresets, fetchTags, fetchTagsIndex } from '../store/slices/tag';
import { CUR_MONTH, CalMonth } from '../utils/DateTime';
import { TrxnNav } from '../components/Trxn/TrxnNav';
import { TrxnDetail } from './Trxn/TrxnDetail';
import { TrxnCombined } from './Trxn/TrxnCombined';
import { TrxnGraphic } from './Trxn/TrxnGraphic';

export enum ViewMode {
  Detail, Combined, Graph
};

function TrxnMain() {
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.Detail);
  const [curMonth, setCurMonth] = useState<CalMonth>(CUR_MONTH);

  const dispatch = useDispatch<AppDispatch>();
  const { errorState } = useSelector(selectTrxn);

  useEffect(() => {
    dispatch(fetchTrxns({yearMonth: curMonth}));
    dispatch(fetchCombinedTrxns({yearMonth: curMonth}));
  }, [errorState, curMonth, dispatch]);

  // Fetch Tag Related Things!
  useEffect(() => {
    dispatch(fetchTags());
    dispatch(fetchTagsIndex());
    dispatch(fetchTagPresets());
  }, [dispatch]);

  return (
    <div className="App">
      <TrxnInput />
      <TrxnNav viewMode={viewMode} setViewMode={setViewMode} curMonth={curMonth} setCurMonth={setCurMonth} />
      {viewMode === ViewMode.Detail && <TrxnDetail curMonth={curMonth} />}
      {viewMode === ViewMode.Combined && <TrxnCombined curMonth={curMonth} />}
      {viewMode === ViewMode.Graph && <TrxnGraphic curMonth={curMonth} />}
    </div>
  );
}

export default TrxnMain;