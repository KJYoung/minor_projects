import React, { useState } from "react";

const Auth = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [newAccount, setNewAccount] = useState(false);
  
  const onSubmit = (event) => {
    event.preventDefault();
  };
  return (
    <div>
        <form onSubmit={onSubmit}>
            <input name="email" type="text" placeholder="Email" required
                   value={email} onChange={(e) => setEmail(e.target.value)}/>
            <input name="password" type="password" placeholder="Password" required
                   value={password} onChange={(e) => setPassword(e.target.value)}/>
            <input type="submit" value="Log In" />
        </form>
        <div>
            <button>Continue with Google</button>
            <button>Continue with Github</button>
        </div>
    </div>)
  };
export default Auth;