/* eslint-disable @typescript-eslint/no-unused-vars */
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { AxiosError, AxiosResponse } from 'axios';
import { put, call, takeLatest } from 'redux-saga/effects';
import * as CoreAPI from '../apis/core';

interface CoreState {
  id: number,
  name: string
}

export const initialState: CoreState = {
  id: 0,
  name: ''
};

export const coreSlice = createSlice({
  name: 'core',
  initialState,
  reducers: {
    stateRefresh: () => initialState,

    getElements: state => {
        state.id = 0;
        state.name = '';
    },
    getCoresSuccess: (state, { payload }) => {
        state.id = 10;
        state.name = 'hi';
    },
    getCoresFailure: (state, { payload }) => {
        state.id = -1;
        state.name = 'ERROR';
    },

    createCore: (state, action: PayloadAction<CoreAPI.CorePostReqType>) => {
    },
    createCoreSuccess: (state, { payload }) => {
    },
    createCoreFailure: (state, { payload }) => {
    },
  },
});
export const coreActions = coreSlice.actions;

function* getCoresSaga() {
  try {
    const response: AxiosResponse = yield call(CoreAPI.getElements);
    yield put(coreActions.getCoresSuccess(response));
  } catch (error) {
    yield put(coreActions.getCoresFailure(error));
  }
}
function* createCoreSaga(action: PayloadAction<CoreAPI.CorePostReqType>) {
  try {
    const response: AxiosResponse = yield call(CoreAPI.postElement, action.payload);
    yield put(coreActions.getCoresSuccess(response));
  } catch (error) {
    yield put(coreActions.createCoreFailure(error));
  }
}

export default function* CoreSaga() {
  yield takeLatest(coreActions.getElements, getCoresSaga);
  yield takeLatest(coreActions.createCore, createCoreSaga);
}
