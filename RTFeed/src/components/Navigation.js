import React from "react";
import { useNavigate } from "react-router-dom";

const Navigation = () => {
    const navigate = useNavigate();

    return <nav>
        <ul>
            <li><span onClick={() => navigate("/")}>Home</span></li>
            <li><span onClick={() => navigate("/profile")}>MyProfile</span></li>
        </ul>
    </nav>
}


export default Navigation;