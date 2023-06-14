/* eslint-disable @typescript-eslint/no-unused-vars */
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { AxiosError, AxiosResponse } from 'axios';
import * as CoreAPI from '../apis/core';

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

export const coreSlice = createSlice({
  name: 'core',
  initialState,
  reducers: {
    stateRefresh: () => initialState,

    getElements: state => {
        state.elements = [];
        state.errorState = ERRORSTATE.DEFAULT;
    },
    getCoresSuccess: (state, { payload }) => {
      state.elements = payload.elements;
      state.errorState = ERRORSTATE.NORMAL;
    },
    getCoresFailure: (state, { payload }) => {
      state.elements = [];
      state.errorState = ERRORSTATE.ERROR;
    },
    
    createCore: (state, action: PayloadAction<CoreAPI.CorePostReqType>) => {
      state.errorState = ERRORSTATE.DEFAULT;
    },
    createCoreSuccess: (state, { payload }) => {
      state.errorState = ERRORSTATE.SUCCESS;
    },
    createCoreFailure: (state, { payload }) => {
      state.errorState = ERRORSTATE.ERROR;
    },
    
    editCore: (state, action: PayloadAction<CoreAPI.CorePutReqType>) => {
      state.errorState = ERRORSTATE.DEFAULT;
    },
    editCoreSuccess: (state, { payload }) => {
      state.errorState = ERRORSTATE.SUCCESS;
    },
    editCoreFailure: (state, { payload }) => {
      state.errorState = ERRORSTATE.ERROR;
    },

    deleteCore: (state, action: PayloadAction<CoreAPI.CoreDeleteReqType>) => {
      state.errorState = ERRORSTATE.DEFAULT;
    },
    deleteCoreSuccess: (state, { payload }) => {
      state.errorState = ERRORSTATE.SUCCESS;
    },
    deleteCoreFailure: (state, { payload }) => {
      state.errorState = ERRORSTATE.ERROR;
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
    yield put(coreActions.createCoreSuccess(response));
  } catch (error) {
    yield put(coreActions.createCoreFailure(error));
  }
}
function* editCoreSaga(action: PayloadAction<CoreAPI.CorePutReqType>) {
  try {
    const response: AxiosResponse = yield call(CoreAPI.putElement, action.payload);
    yield put(coreActions.editCoreSuccess(response));
  } catch (error) {
    yield put(coreActions.editCoreFailure(error));
  }
}
function* deleteCoreSaga(action: PayloadAction<CoreAPI.CoreDeleteReqType>) {
  try {
    const response: AxiosResponse = yield call(CoreAPI.deleteElement, action.payload);
    yield put(coreActions.deleteCoreSuccess(response));
  } catch (error) {
    yield put(coreActions.deleteCoreFailure(error));
  }
}

export default function* CoreSaga() {
  yield takeLatest(coreActions.getElements, getCoresSaga);
  yield takeLatest(coreActions.createCore, createCoreSaga);
  yield takeLatest(coreActions.editCore, editCoreSaga);
  yield takeLatest(coreActions.deleteCore, deleteCoreSaga);
}
