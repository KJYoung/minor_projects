/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction, current } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TagElement } from './tag';
import { CalMonth } from '../../utils/DateTime';

export type TrxnElement = {
  id: number,
  date: string,
  memo: string,
  tag: TagElement[],
  period: number,
  amount: number,
};

export interface TrxnCreateReqType {
  date: string,
  memo: string,
  tag: TagElement[],
  period: number,
  amount: number,
};
export interface TrxnFetchReqType {
  searchKeyword?: string, // 검색 키워드.
  yearMonth?: CalMonth, // 년/월 정보.
  // fetchSize?: number, // 가져오는 Transaction 개수.
};
export interface CombinedTrxnFetchReqType {
  yearMonth: CalMonth, // 년/월 정보.
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
  rawElements: TrxnElement[], // Raw Fetched Data From Backend
  elements: TrxnElement[], // Sorted, Filtered Data in Frontend
  combined: Number[], // Combined Daily Amount
  errorState: ERRORSTATE,
  sortState: TrxnSortState,
  filterTag: TagElement[],
};

export const initialState: TrxnState = {
  rawElements: [], 
  elements: [],
  combined: [],
  errorState: ERRORSTATE.DEFAULT,
  sortState: defaultTrxnSortState,
  filterTag: [],
};

export const fetchTrxns = createAsyncThunk(
  "trxn/fetchTrxns",
  async (payload: TrxnFetchReqType) => {
    let reqLink = "/api/trxn/?combined=False";

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

export const fetchCombinedTrxns = createAsyncThunk(
  "trxn/fetchCombinedTrxns",
  async (payload: CombinedTrxnFetchReqType) => {
    let reqLink = "/api/trxn/combined/";

    if(payload.yearMonth.month)
      reqLink = `${reqLink}?year=${payload.yearMonth.year}&month=${payload.yearMonth.month}`;
    else
      reqLink = `${reqLink}?year=${payload.yearMonth.year}`;

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

const trxnSortFn = (trxnList: TrxnElement[], trxnSortState: TrxnSortState, filterTag: TagElement[]) : TrxnElement[] => {
  // Date Sort
  if(trxnSortState.date === SortState.Ascend)
    trxnList.sort((a, b) => a.date.localeCompare(b.date));
  if(trxnSortState.date === SortState.Descend)
    trxnList.sort((a, b) => b.date.localeCompare(a.date));
  
  // Period Sort
  if(trxnSortState.period === SortState.Ascend)
    trxnList.sort((a, b) => a.period - b.period);
  if(trxnSortState.period === SortState.Descend)
    trxnList.sort((a, b) => b.period - a.period);

  // Tag Filter...
  if(trxnSortState.tag === SortState.TagFilter){
    filterTag.forEach((tag) => {
      trxnList = trxnList.filter((trxn) => {
        return trxn.tag.find((trxnTag) => trxnTag.id === tag.id) !== undefined;
      });
    });
  };

  // Amount Sort
  if(trxnSortState.amount === SortState.Ascend)
    trxnList.sort((a, b) => a.amount - b.amount);
  if(trxnSortState.amount === SortState.Descend)
    trxnList.sort((a, b) => b.amount - a.amount);

  // Memo Sort
  if(trxnSortState.memo === SortState.Ascend)
    trxnList.sort((a, b) => a.memo.localeCompare(b.memo));
  if(trxnSortState.memo === SortState.Descend)
    trxnList.sort((a, b) => b.memo.localeCompare(a.memo));
  
  return trxnList;
};


export const TrxnSlice = createSlice({
  name: "trxn",
  initialState,
  reducers: {
    setTrxnSort: (state, action: PayloadAction<TrxnSortState>) => {
      state.sortState = action.payload;
      state.elements = trxnSortFn([...state.rawElements], state.sortState, state.filterTag); // Sort!
    },
    clearTrxnSort: (state, action: PayloadAction<{}>) => {
      state.sortState = defaultTrxnSortState;
    },
    setTrxnFilterTag: (state, action: PayloadAction<TagElement[]>) => {
      state.filterTag = action.payload;
      state.elements = trxnSortFn([...state.rawElements], state.sortState, state.filterTag); // Sort!
    },
  },
  extraReducers(builder) {
    builder.addCase(fetchTrxns.fulfilled, (state, action) => {
      state.rawElements = action.payload.elements;
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    });
    builder.addCase(fetchCombinedTrxns.fulfilled, (state, action) => {
      console.log(action.payload);
      state.combined = action.payload.elements;
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