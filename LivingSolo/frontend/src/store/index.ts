import { configureStore } from '@reduxjs/toolkit';
import trxnReducer from "./slices/trxn";
import tagReducer from "./slices/tag";
import todoReducer from "./slices/todo";

export const store = configureStore({ 
    reducer : { 
        trxn: trxnReducer,
        tag: tagReducer,
        todo: todoReducer,
    }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;