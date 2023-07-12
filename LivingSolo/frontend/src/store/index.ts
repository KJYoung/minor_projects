import { configureStore } from '@reduxjs/toolkit';
import coreReducer from "./slices/core";
import trxnReducer from "./slices/trxn";
import trxnTypeReducer from "./slices/trxnType";

export const store = configureStore({ 
    reducer : { 
        core: coreReducer,
        trxn: trxnReducer,
        trxnType: trxnTypeReducer,
    }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;