import { authCreateUser, authLogIn, authService } from "fbConfig";
import React, { useState } from "react";

const AuthForm = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [newAccount, setNewAccount] = useState(true);

    const toggleAccount = () => setNewAccount(nA => !nA);
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

    return <>
        <form onSubmit={onSubmit}>
            <input name="email" type="text" placeholder="Email" required
                value={email} onChange={(e) => setEmail(e.target.value)}/>
            <input name="password" type="password" placeholder="Password" required
                value={password} onChange={(e) => setPassword(e.target.value)}/>
            <input type="submit" value={newAccount ? "Create Account" : "Log In"} />
            <span>{error}</span>
        </form>
        <span onClick={toggleAccount}>{newAccount ? "로그인" : "회원가입"}</span>
    </>
};

export default AuthForm;