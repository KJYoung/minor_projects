import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import createSagaMiddleware from 'redux-saga';
import { configureStore } from '@reduxjs/toolkit';
import { rootReducer, rootSaga } from './store';
import App from './App';

const sagaMiddleware = createSagaMiddleware();
export const store = configureStore({
  reducer: rootReducer,
  middleware: [sagaMiddleware],
  devTools: process.env.REACT_APP_MODE === 'development',
});
sagaMiddleware.run(rootSaga);
document.getElementById('root')?.setAttribute('spellcheck', 'false');

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <Provider store={store}>
    <App />
  </Provider>,
);

export type RootState = ReturnType<typeof rootReducer>;
