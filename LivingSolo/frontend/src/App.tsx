import React from 'react';
import { useDispatch } from 'react-redux';
import { coreActions } from './store/slices/core';

function App() {
  const dispatch = useDispatch();

  return (
    <div className="App">
      <button onClick={() => {
        dispatch(coreActions.getElements());
      }}>
          CLICK!
      </button>
    </div>
  );
}

export default App;
