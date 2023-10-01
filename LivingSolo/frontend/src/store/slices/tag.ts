/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TrxnElement } from './trxn';
import { TodoCategory, TodoElement } from './todo';
import { IDReqType } from '../../utils/Interfaces';

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

export type TagPreset = {
  id: number,
  name: string,
  tags: TagElement[],
};

export type TagDetail = {
  transaction: TrxnElement[],
  todo: TodoElement[],
};

interface TagState {
  elements: TagClassElement[],
  index: TagElement[],
  preset: TagPreset[],
  errorState: ERRORSTATE,
  tagDetail: TagDetail | undefined,
};

export const initialState: TagState = {
  elements: [],
  index: [],
  preset: [],
  errorState: ERRORSTATE.DEFAULT,
  tagDetail: undefined,
};

export const fetchTags = createAsyncThunk(
  "tag/fetchTags",
  async () => {
    const response = await client.get(`/api/tag/class/`);
    return response.data;
  }
);
export const fetchTagDetail = createAsyncThunk(
  "tag/fetchTagDetail",
  async (id: number) => {
    const response = await client.get(`/api/tag/${id}/`);
    return response.data;
  }
);
export const fetchTagsIndex = createAsyncThunk(
  "tag/fetchTagsIndex",
  async () => {
    const response = await client.get(`/api/tag/`);
    return response.data;
  }
);
export const fetchTagPresets = createAsyncThunk(
  "tag/fetchTagPresets",
  async () => {
    const response = await client.get(`/api/tag/preset/`);
    return response.data;
  }
);
// Post
export interface TagClassCreateReqType {
  name: string,
  color: string,
};
export interface TagCreateReqType {
  name: string,
  color: string,
  class: string,
};
export interface TagPresetReqType {
  name: string,
  tags: TagElement[],
};

export const createTagClass = createAsyncThunk(
  "tag/createTagCategory",
  async (payload: TagClassCreateReqType) => {
    const response = await client.post(`/api/tag/class/`, payload);
    return response.data;
  }
);
export const createTag = createAsyncThunk(
  "tag/createTag",
  async (payload: TagCreateReqType) => {
    const response = await client.post(`/api/tag/`, payload);
    return response.data;
  }
);
export const createTagPreset = createAsyncThunk(
  "tag/createTagPreset",
  async (payload: TagPresetReqType) => {
    const response = await client.post(`/api/tag/preset/`, payload);
    return response.data;
  }
);

// Put

// Delete
export const deleteTag = createAsyncThunk(
  "tag/deleteTag",
  async (payload: IDReqType) => {
    const response = await client.delete(`/api/tag/${payload.id}`);
    return response.data;
  }
)
export const deleteTagPreset = createAsyncThunk(
  "tag/deleteTagPreset",
  async (payload: IDReqType) => {
    const response = await client.delete(`/api/tag/preset/${payload.id}`);
    return response.data;
  }
)

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
    builder.addCase(fetchTagPresets.fulfilled, (state, action) => {
      state.preset = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    }); 
    builder.addCase(fetchTagDetail.fulfilled, (state, action) => {
      state.tagDetail = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    }); 
    [
      createTagClass, createTag, createTagPreset,
      deleteTag, deleteTagPreset
    ].forEach((reducer) => {
      builder.addCase(reducer.pending, (state, action) => {
        state.errorState = ERRORSTATE.DEFAULT;
      });
      builder.addCase(reducer.fulfilled, (state, action) => {
        state.errorState = ERRORSTATE.SUCCESS;
      });
    });
  },
});

export const TagActions = TagSlice.actions;
export const selectTag = (state: RootState) => state.tag;
export default TagSlice.reducer;