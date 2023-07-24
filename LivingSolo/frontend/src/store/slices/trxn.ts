/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TagBubbleElement } from './tag';
import { CalMonth } from '../../utils/DateTime';

export type TrxnElement = {
  id: number,
  date: string,
  memo: string,
  tag: TagBubbleElement[],
  period: number,
  amount: number,
};

export interface TrxnCreateReqType {
  date: string,
  memo: string,
  tag: TagBubbleElement[],
  period: number,
  amount: number,
};
export interface TrxnFetchReqType {
  dayCombined?: boolean, // 일별 거래로 뭉쳐서 요청.
  searchKeyword?: string, // 검색 키워드.
  yearMonth?: CalMonth, // 년/월 정보.
  // fetchSize?: number, // 가져오는 Transaction 개수.
};

export enum SortState {
  NotSort, Ascend, Descend, TagFilter
};
export enum TrxnSortTarget {
  Date, Period, Tag, Amount, Memo
};

export interface TrxnSortState {
  date: SortState.NotSort | SortState.Ascend | SortState.Descend,
  period: SortState.NotSort | SortState.Ascend | SortState.Descend,
  tag: SortState.NotSort | SortState.TagFilter,
  amount: SortState.NotSort | SortState.Ascend | SortState.Descend,
  memo: SortState.NotSort | SortState.Ascend | SortState.Descend,
};

export const defaultTrxnSortState : TrxnSortState = {
  date: SortState.NotSort,
  period: SortState.NotSort,
  tag: SortState.NotSort,
  amount: SortState.NotSort,
  memo: SortState.NotSort,
};

interface TrxnState {
  elements: TrxnElement[],
  errorState: ERRORSTATE,
  sortState: TrxnSortState,
};

export const initialState: TrxnState = {
  elements: [],
  errorState: ERRORSTATE.DEFAULT,
  sortState: defaultTrxnSortState
};

export const fetchTrxns = createAsyncThunk(
  "trxn/fetchTrxns",
  async (payload: TrxnFetchReqType) => {
    let reqLink = "/api/trxn/";

    if(payload.dayCombined){
      reqLink = `${reqLink}?combined=True`
    }else{
      reqLink = `${reqLink}?combined=False`
    };

    if(payload.yearMonth){
      if(payload.yearMonth.month)
        reqLink = `${reqLink}&year=${payload.yearMonth.year}&month=${payload.yearMonth.month}`;
      else
        reqLink = `${reqLink}&year=${payload.yearMonth.year}`;
    };

    if(payload.searchKeyword){
      reqLink = `${reqLink}&keyword=${payload.searchKeyword}`
    };

    const response = await client.get(reqLink);
    return response.data;
  }
);
export const createTrxn = createAsyncThunk(
  "trxn/createTrxn",
  async (trxnCreateObj: TrxnCreateReqType, { dispatch }) => {
    await client.post("/api/trxn/", trxnCreateObj);
  }
);
export const editTrxn = createAsyncThunk(
  "trxn/editTrxn",
  async (editTrxnObj: TrxnElement, { dispatch }) => {
    await client.put(`/api/trxn/${editTrxnObj.id}/`, editTrxnObj);
  }
);
export const deleteTrxn = createAsyncThunk(
  "trxn/deleteTrxn",
  async (TrxnID: String | Number, { dispatch }) => {
    await client.delete(`/api/trxn/${TrxnID}/`);
  }
);

export const TrxnSlice = createSlice({
  name: "trxn",
  initialState,
  reducers: {
    setTrxnSort: (state, action: PayloadAction<TrxnSortState>) => {
      state.sortState = action.payload;
    },
    clearTrxnSort: (state, action: PayloadAction<{}>) => {
      state.sortState = defaultTrxnSortState;
    },
  },
  extraReducers(builder) {
    builder.addCase(fetchTrxns.fulfilled, (state, action) => {
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    });
    builder.addCase(createTrxn.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
    builder.addCase(editTrxn.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
    builder.addCase(deleteTrxn.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
  },
});

export const TrxnActions = TrxnSlice.actions;
export const selectTrxn = (state: RootState) => state.trxn;
export default TrxnSlice.reducer;