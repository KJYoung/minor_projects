import AuthForm from "components/AuthForm";
import React from "react";
import { authService, authGoogleProvider, authGithubProvider, authSignUpWithPopUp } from "../fbConfig";

const Auth = () => {
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
        <AuthForm />
        <div>
            <button name="google" onClick={onSocialLogin}>Continue with Google</button>
            <button name="github" onClick={onSocialLogin}>Continue with Github</button>
        </div>
    </div>)
  };
export default Auth;