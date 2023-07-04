import { configureStore } from '@reduxjs/toolkit';
import coreReducer from "./slices/core";
import trxnReducer from "./slices/trxn";

export const store = configureStore({ 
    reducer : { 
        core: coreReducer,
        trxn: trxnReducer,
    }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;