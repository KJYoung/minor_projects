import { useEffect, useState } from "react";
import MyRouter from "components/Router";
import { authService } from "fbConfig";
import { onAuthChange } from "../fbConfig";

function App() {
  const [init, setInit] = useState(false);
  const [userObj, setUserObj] = useState(null);

  useEffect(() => {
    onAuthChange(authService, (user) => {
      if(user){
        setUserObj(user);
      }else{
        setUserObj(null);
      }
      setInit(true);
    });
  }, []);
  return (
    <>
      {init ? <MyRouter isLoggedIn={userObj !== null} userObj={userObj} /> : "Loading Firebase..."}
      <footer>&copy; RTFeed. 2023 Jan.</footer>
    </>
  );
}

export default App;
