/* eslint-disable @typescript-eslint/no-unused-vars */
import { createAsyncThunk, createSlice, PayloadAction, current } from '@reduxjs/toolkit';
import client from '../apis/client';
import { RootState } from '..';
import { ERRORSTATE } from './core';
import { TagElement } from './tag';
import { CalMonth } from '../../utils/DateTime';

export type TodoCategory = {
  id: number,
  name: string,
  color: string,
  tag: TagElement[],
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
export interface TodoToggleDoneReqType {
  id: number,
};
export interface TodoCreateReqType {
  name: string,
  tag: TagElement[],
  category: string,
  priority: number,
  deadline: string,
  is_hard_deadline: boolean,
  period: number,
};
export interface TodoCategoryCreateReqType {
  name: string,
  tag: TagElement[],
  color: string,
};

interface TodoState {
  elements: (TodoElement[])[],
  categories: TodoCategory[],
  errorState: ERRORSTATE,
};

export const initialState: TodoState = {
  elements: [],
  categories: [],
  errorState: ERRORSTATE.DEFAULT,
};

export const fetchTodos = createAsyncThunk(
  "todo/fetchTodos",
  async (payload: TodoFetchReqType) => {
    const reqLink = `/api/todo/?&year=${payload.yearMonth.year}&month=${payload.yearMonth.month! + 1}`;
    const response = await client.get(reqLink);
    return response.data;
  }
);
export const fetchTodoCategory = createAsyncThunk(
  "todo/fetchTodoCategory",
  async () => {
    const response = await client.get(`/api/todo/category/`);
    return response.data;
  }
);
export const toggleTodoDone = createAsyncThunk(
  "todo/toggleTodoDone",
  async (payload: TodoToggleDoneReqType) => {
    const response = await client.put(`/api/todo/toggle/${payload.id}/`);
    return response.data;
  }
);
export const createTodo = createAsyncThunk(
  "todo/createTodo",
  async (payload: TodoCreateReqType) => {
    const response = await client.post(`/api/todo/`, payload);
    return response.data;
  }
);
export const createTodoCategory = createAsyncThunk(
  "todo/createTodoCategory",
  async (payload: TodoCategoryCreateReqType) => {
    const response = await client.post(`/api/todo/category/`, payload);
    return response.data;
  }
);
export const deleteTodoCategory = createAsyncThunk(
  "todo/deleteTodoCategory",
  async (todoCategoryID: String | Number, { dispatch }) => {
    await client.delete(`/api/todo/category/${todoCategoryID}/`);
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
    builder.addCase(fetchTodoCategory.fulfilled, (state, action) => {
      state.categories = action.payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    });
    builder.addCase(toggleTodoDone.pending, (state, action) => {
      state.errorState = ERRORSTATE.DEFAULT;
    });
    builder.addCase(toggleTodoDone.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    });
    builder.addCase(createTodo.pending, (state, action) => {
      state.errorState = ERRORSTATE.DEFAULT;
    });
    builder.addCase(createTodo.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    });
    builder.addCase(createTodoCategory.pending, (state, action) => {
      state.errorState = ERRORSTATE.DEFAULT;
    });
    builder.addCase(createTodoCategory.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    });
    builder.addCase(deleteTodoCategory.pending, (state, action) => {
      state.errorState = ERRORSTATE.DEFAULT;
    });
    builder.addCase(deleteTodoCategory.fulfilled, (state, action) => {
      state.errorState = ERRORSTATE.SUCCESS;
    });
  },
});

export const TodoActions = TodoSlice.actions;
export const selectTodo = (state: RootState) => state.todo;
export default TodoSlice.reducer;