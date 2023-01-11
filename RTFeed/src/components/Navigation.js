import React from "react";
import { useNavigate } from "react-router-dom";

const Navigation = ({ userObj }) => {
    const navigate = useNavigate();

    return <nav>
        <ul>
            <li><span onClick={() => navigate("/")}>Home</span></li>
            <li><span onClick={() => navigate("/profile")}>{userObj.displayName}의 프로필</span></li>
        </ul>
    </nav>
}


export default Navigation;