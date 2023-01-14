import { faGithub, faGoogle, faTwitter } from "@fortawesome/free-brands-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
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
    <div className="authContainer">
        <FontAwesomeIcon icon={faTwitter} color={"#04AAFF"} size="3x" style={{ marginBottom: 30 }} />
        <AuthForm />
        <div className="authBtns">
            <button name="google" onClick={onSocialLogin} className="authBtn">
              Continue with Google <FontAwesomeIcon icon={faGoogle} />
            </button>
            <button name="github" onClick={onSocialLogin} className="authBtn">
              Continue with Github <FontAwesomeIcon icon={faGithub} />
            </button>
        </div>
    </div>)
  };
export default Auth;