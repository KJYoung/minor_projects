import { useEffect, useState } from "react";
import MyRouter from "components/Router";
import { authService, authUpdateProfile } from "fbConfig";
import { onAuthChange } from "../fbConfig";

function App() {
  const [init, setInit] = useState(false);
  const [userObj, setUserObj] = useState(null);

  useEffect(() => {
    onAuthChange(authService, (user) => {
      if(user){
        setUserObj({
          displayName: user.displayName,
          uid: user.uid,
          updateProfile: (name) => authUpdateProfile(user, { displayName: name }),
        });
      }else{
        setUserObj(null);
      }
      setInit(true);
    });
  }, []);
  const refreshUser = () => {
    const user = authService.currentUser;
    setUserObj({
      displayName: user.displayName,
      uid: user.uid,
      updateProfile: (name) => authUpdateProfile(user, { displayName: name }),
    });
  };
  return (
    <>
      {init ? <MyRouter refreshUser={refreshUser} isLoggedIn={userObj !== null} userObj={userObj} /> : "Loading Firebase..."}
      <footer>&copy; RTFeed. 2023 Jan.</footer>
    </>
  );
}

export default App;
