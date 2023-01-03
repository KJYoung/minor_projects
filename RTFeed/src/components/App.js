import { useState } from "react";
import MyRouter from "components/Router";
import { authService } from "fbConfig";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(authService.currentUser);
  return (
    <>
      <MyRouter isLoggedIn={isLoggedIn} />
      <footer>&copy; RTFeed. 2023 Jan.</footer>
    </>
  );
}

export default App;
