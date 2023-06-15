/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import axios, { AxiosError, AxiosResponse } from 'axios';
import * as CoreAPI from '../apis/core';
import client from '../apis/client';
import { RootState } from '..';

export enum ERRORSTATE {
  DEFAULT, NORMAL, SUCCESS, PENDING, ERROR
  // DESIGN CHOICE: Normal is for the SUCCESS of the get Elements.
};

type CoreElement = {
  id: number,
  name: string
};

interface CoreState {
  elements: CoreElement[],
  errorState: ERRORSTATE,
};

export const initialState: CoreState = {
  elements: [],
  errorState: ERRORSTATE.DEFAULT
};

export const fetchCore = createAsyncThunk(
  "core/fetchCore",
  async () => {
    const response = await client.get(`/api/core/`);
    return response.data;
  }
)

export const coreSlice = createSlice({
  name: "core",
  initialState,
  reducers: {
    getElement: (state, action: PayloadAction<{}>) => {},
  }
});

export const coreActions = coreSlice.actions;
export const selectCore = (state: RootState) => state.core;
export default coreSlice.reducer;