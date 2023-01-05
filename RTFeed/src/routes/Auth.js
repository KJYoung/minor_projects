import React, { useState } from "react";
import { authService, authCreateUser, authLogIn, authGoogleProvider, authGithubProvider, authSignUpWithPopUp } from "../fbConfig";

const Auth = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [newAccount, setNewAccount] = useState(true);
  const [error, setError] = useState("");

  const onSubmit = async (event) => {
    event.preventDefault();
    try{
        if(newAccount){
            // Create Account
            await authCreateUser(authService, email, password);
        }else{
            // Log In
            await authLogIn(authService, email, password);
        }
    }catch(error){
        setError(error.message);
    }
  };
  const toggleAccount = () => setNewAccount(nA => !nA);
  const onSocialLogin = async (event) => {
    const {target : {name}} = event;
    let provider;
    switch(name){
        case "google":
            provider = new authGoogleProvider();
            break;
        case "github":
            provider = new authGithubProvider();
            break;
        default:
            return;
    }
    await authSignUpWithPopUp(authService, provider);
  }
  return (
    <div>
        <form onSubmit={onSubmit}>
            <input name="email" type="text" placeholder="Email" required
                   value={email} onChange={(e) => setEmail(e.target.value)}/>
            <input name="password" type="password" placeholder="Password" required
                   value={password} onChange={(e) => setPassword(e.target.value)}/>
            <input type="submit" value={newAccount ? "Create Account" : "Log In"} />
            <span>{error}</span>
        </form>
        <span onClick={toggleAccount}>{newAccount ? "로그인" : "회원가입"}</span>
        <div>
            <button name="google" onClick={onSocialLogin}>Continue with Google</button>
            <button name="github" onClick={onSocialLogin}>Continue with Github</button>
        </div>
    </div>)
  };
export default Auth;