import { configureStore } from '@reduxjs/toolkit';
import coreReducer from "./slices/core";
import trxnReducer from "./slices/trxn";
import tagReducer from "./slices/tag";
import todoReducer from "./slices/todo";

export const store = configureStore({ 
    reducer : { 
        core: coreReducer,
        trxn: trxnReducer,
        tag: tagReducer,
        todo: todoReducer,
    }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;