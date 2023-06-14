import { combineReducers } from '@reduxjs/toolkit';
import coreSaga, { coreSlice } from './slices/core';

export const rootReducer = combineReducers({
  core: coreSlice.reducer,
});
export function* rootSaga() {
  yield all([fork(coreSaga)]);
}
