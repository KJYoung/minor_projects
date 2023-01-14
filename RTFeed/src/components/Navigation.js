import { faTwitter } from "@fortawesome/free-brands-svg-icons";
import { faUser } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React from "react";
import { useNavigate } from "react-router-dom";

const Navigation = ({ userObj }) => {
    const navigate = useNavigate();

    return <nav>
        <ul style={{ display: "flex", justifyContent: "center", marginTop: 50 }}>
            <li>
                <span onClick={() => navigate("/")} style={{ marginRight: 10 }}>
                    <FontAwesomeIcon icon={faTwitter} color={"#04AAFF"} size="2x" />
                </span>
            </li>
            <li>
                <span onClick={() => navigate("/profile")} style={{
                    marginLeft: 10,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    fontSize: 12,
                }}>
                    <FontAwesomeIcon icon={faUser} color={"#04AAFF"} size="2x" />
                    <span style={{ marginTop: 10}}>{userObj.displayName}의 프로필</span>
                </span>
            </li>
        </ul>
    </nav>
}


export default Navigation;