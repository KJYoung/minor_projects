/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TypeBubbleElement } from './trxnType';

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
  async () => {
    const response = await client.get(`/api/trxn/`);
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