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

export const fetchCores = createAsyncThunk(
  "core/fetchCores",
  async () => {
    const response = await client.get(`/api/core/`);
    return response.data;
  }
);
export const createCore = createAsyncThunk(
  "core/createCore",
  async (coreName: String, { dispatch }) => {
      const response = await client.post("/api/core/", {name : coreName});
      dispatch(coreActions.createCore(response.data));
  }
);

export const editCore = createAsyncThunk(
  "core/editCore",
  async (editCoreObj: CoreElement, { dispatch }) => {
      const response = await client.put(`/api/core/${editCoreObj.id}/`, {name : editCoreObj.name});
      dispatch(coreActions.editCore(response.data));
  }
);
export const deleteCore = createAsyncThunk(
  "core/deleteCore",
  async (coreID: String | Number, { dispatch }) => {
      const response = await client.delete(`/api/core/${coreID}/`);
      dispatch(coreActions.deleteCore(response.data));
  }
);

export const coreSlice = createSlice({
  name: "core",
  initialState,
  reducers: {
    createCore: (state, action: PayloadAction<{}>) => {},
    editCore: (state, action: PayloadAction<{}>) => {},
    deleteCore: (state, action: PayloadAction<{}>) => {},
  },
  extraReducers(builder) {
    builder.addCase(fetchCores.fulfilled, (state, action) => {
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.DEFAULT;
    }); 
    builder.addCase(createCore.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
    builder.addCase(editCore.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
    builder.addCase(deleteCore.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    }); 
  },
});

export const coreActions = coreSlice.actions;
export const selectCore = (state: RootState) => state.core;
export default coreSlice.reducer;