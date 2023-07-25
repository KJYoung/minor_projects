/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';

export type TagClassElement = {
  id: number,
  name: string,
  color: string,
  tags?: TagElement[]
};

// without tag_class, TagElement is "TagBubbleElement"
export type TagElement = {
  id: number,
  name: string,
  color: string,
  tag_class?: TagElement
};


interface TagState {
  elements: TagClassElement[],
  index: TagElement[],
  errorState: ERRORSTATE,
};

export const initialState: TagState = {
  elements: [],
  index: [],
  errorState: ERRORSTATE.DEFAULT
};

export const fetchTags = createAsyncThunk(
  "trxn/fetchTags",
  async () => {
    const response = await client.get(`/api/tag/class/`);
    return response.data;
  }
);
export const fetchTagsIndex = createAsyncThunk(
  "trxn/fetchTagsIndex",
  async () => {
    const response = await client.get(`/api/tag/`);
    return response.data;
  }
);

export const TagSlice = createSlice({
  name: "tag",
  initialState,
  reducers: {},
  extraReducers(builder) {
    builder.addCase(fetchTags.fulfilled, (state, action) => {
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    }); 
    builder.addCase(fetchTagsIndex.fulfilled, (state, action) => {
      state.index = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    }); 
  },
});

export const TagActions = TagSlice.actions;
export const selectTag = (state: RootState) => state.tag;
export default TagSlice.reducer;