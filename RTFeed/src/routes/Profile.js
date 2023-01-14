import { authLogOut, authService, dbCollection, dbGetDocs, dbOrderBy, dbQuery, dbService, dbWhere } from "fbConfig";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const Profile = ({ userObj, refreshUser }) => {
    const [dpName, setDPName] = useState(userObj.displayName);
    const navigate = useNavigate();
    const onLogOut = async () => {
        await authLogOut(authService);
        navigate("/");
    }
    const getOwnTweets = async () => {
        const q = dbQuery(
            dbCollection(dbService, "tweets"),
            dbWhere("author_uid", "==", userObj.uid),
            dbOrderBy("createdAt", "desc")
        );
        const qSnapshot = await dbGetDocs(q);
        qSnapshot.forEach((doc) => {
            // console.log(doc.id, " => ", doc.data());
        });
    };
    useEffect(() => {
        getOwnTweets();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return <div className="container">
        <form onSubmit={async (e) => {
            e.preventDefault();
            if(userObj.displayName !== dpName){
                // Name Change Occurred!
                await userObj.updateProfile(dpName);
                // Maybe we have to refresh to reflect changes immediately.
                refreshUser();
            }
        }} className="profileForm">
            <input type="text" placeholder="Display name"
                   value={dpName} onChange={e => setDPName(e.target.value)}
                   autoFocus required className="formInput"/>
            <input type="submit" value="Update Name" className="formBtn" style={{ marginTop: 10 }}/>
        </form>
        <span onClick={onLogOut} className="formBtn cancelBtn logOut">Log Out</span>
    </div>;
}
export default Profile;