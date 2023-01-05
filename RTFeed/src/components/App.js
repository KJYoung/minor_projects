import { useEffect, useState } from "react";
import MyRouter from "components/Router";
import { authService } from "fbConfig";
import { onAuthChange } from "../fbConfig";

function App() {
  const [init, setInit] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  useEffect(() => {
    onAuthChange(authService, (user) => {
      if(user){
        setIsLoggedIn(true);
      }else{
        setIsLoggedIn(false);
      }
      setInit(true);
    })
  }, []);
  return (
    <>
      {init ? <MyRouter isLoggedIn={isLoggedIn} /> : "Loading Firebase..."}
      <footer>&copy; RTFeed. 2023 Jan.</footer>
    </>
  );
}

export default App;
