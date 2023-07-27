/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction, current } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TagElement } from './tag';
import { CalMonth } from '../../utils/DateTime';

type TodoCategory = {
    name: string,
    color: string
};
export type TodoElement = {
  id: number,
  name: string,
  done: boolean,
  tag: TagElement[],
  color: string,
  category: TodoCategory,
  priority: number,
  deadline: string,
  is_hard_deadline: boolean,
  period: number,
};

export interface TodoFetchReqType {
  yearMonth: CalMonth, // 년/월 정보.
};

interface TodoState {
  elements: (TodoElement[])[], // Sorted, Filtered Data in Frontend
  errorState: ERRORSTATE,
};

export const initialState: TodoState = {
  elements: [],
  errorState: ERRORSTATE.DEFAULT,
};

export const fetchTodos = createAsyncThunk(
  "trxn/fetchTodos",
  async (payload: TodoFetchReqType) => {
    const reqLink = `/api/todo/?&year=${payload.yearMonth.year}&month=${payload.yearMonth.month! + 1}`;
    const response = await client.get(reqLink);
    return response.data;
  }
);

export const TodoSlice = createSlice({
  name: "todo",
  initialState,
  reducers: {
  },
  extraReducers(builder) {
    builder.addCase(fetchTodos.fulfilled, (state, action) => {
      state.elements = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    });
  },
});

export const TodoActions = TodoSlice.actions;
export const selectTodo = (state: RootState) => state.todo;
export default TodoSlice.reducer;