/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TypeBubbleElement } from './trxnType';
import { CalMonth } from '../../utils/DateTime';

export type TrxnElement = {
  id: number,
  date: string,
  memo: string,
  type: TypeBubbleElement[],
  period: number,
  amount: number,
};

export interface TrxnCreateReqType {
  date: string,
  memo: string,
  type: TypeBubbleElement[],
  period: number,
  amount: number,
};
export interface TrxnFetchReqType {
  dayCombined?: boolean, // 일별 거래로 뭉쳐서 요청.
  searchKeyword?: string, // 검색 키워드.
  yearMonth?: CalMonth, // 년/월 정보.
  // fetchSize?: number, // 가져오는 Transaction 개수.
}

interface TrxnState {
  elements: TrxnElement[],
  errorState: ERRORSTATE,
};

export const initialState: TrxnState = {
  elements: [],
  errorState: ERRORSTATE.DEFAULT
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

/**
 * 
 * 
 * let link = `/api/post/?page=${payload.pageNum}&pageSize=${payload.pageSize}`;

  if (payload.searchKeyword) {
    link += `&search=${payload.searchKeyword}`;
  }
  if (payload.tags.length > 0) {
    for (const tag of payload.tags) {
      link += `&tag=${tag.id}`;
    }
  }
  const response = await client.get<getPostsResponseType>(link);
  return response.data;
 * 
 * 
 */



    return response.data;
  }
);
export const createTrxn = createAsyncThunk(
  "trxn/createTrxn",
  async (trxnCreateObj: TrxnCreateReqType, { dispatch }) => {
      const response = await client.post("/api/trxn/", trxnCreateObj);
      dispatch(TrxnActions.createTrxn(response.data));
  }
);
export const editTrxn = createAsyncThunk(
  "trxn/editTrxn",
  async (editTrxnObj: TrxnElement, { dispatch }) => {
      const response = await client.put(`/api/trxn/${editTrxnObj.id}/`, editTrxnObj);
      dispatch(TrxnActions.editTrxn(response.data));
  }
);
export const deleteTrxn = createAsyncThunk(
  "trxn/deleteTrxn",
  async (TrxnID: String | Number, { dispatch }) => {
      const response = await client.delete(`/api/trxn/${TrxnID}/`);
      dispatch(TrxnActions.deleteTrxn(response.data));
  }
);

export const TrxnSlice = createSlice({
  name: "trxn",
  initialState,
  reducers: {
    createTrxn: (state, action: PayloadAction<{}>) => {},
    editTrxn: (state, action: PayloadAction<{}>) => {},
    deleteTrxn: (state, action: PayloadAction<{}>) => {},
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