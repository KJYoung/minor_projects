/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';

export type TypeBubbleElement = {
  id: number,
  name: string,
  color: string
};

export type TrxnTypeClassElement = {
  id: number,
  name: string,
  color: string
};

export type TrxnTypeElement = {
  id: number,
  name: string,
  color: string,
  type_class: TrxnTypeElement
};

interface TrxnTypeState {
  elements: TrxnTypeElement[],
  errorState: ERRORSTATE,
};

export const initialState: TrxnTypeState = {
  elements: [],
  errorState: ERRORSTATE.DEFAULT
};

export const fetchTrxnTypes = createAsyncThunk(
  "trxn/fetchTrxnTypes",
  async () => {
    const response = await client.get(`/api/trxn/type/`);
    return response.data;
  }
);

export const TrxnTypeSlice = createSlice({
  name: "trxnType",
  initialState,
  reducers: {},
  extraReducers(builder) {
    builder.addCase(fetchTrxnTypes.fulfilled, (state, action) => {
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    }); 
  },
});

export const TrxnTypeActions = TrxnTypeSlice.actions;
export const selectTrxnType = (state: RootState) => state.trxnType;
export default TrxnTypeSlice.reducer;