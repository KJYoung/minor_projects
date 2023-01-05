import { authLogOut, authService } from "fbConfig";
import React from "react";
import { useNavigate } from "react-router-dom";

const Profile = () => {
    const navigate = useNavigate();
    const onLogOut = async () => {
        await authLogOut(authService);
        navigate("/");
    }
    return <div>
        <button onClick={onLogOut}>Log Out</button>
        Profile
    </div>;
}
export default Profile;