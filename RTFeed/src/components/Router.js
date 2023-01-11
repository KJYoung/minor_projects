import React from "react";
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import Auth from "routes/Auth";
import Home from "routes/Home";
import Navigation from "components/Navigation";
import Profile from "routes/Profile";

const MyRouter = ({ isLoggedIn, userObj })  => {
    return <Router>
        {isLoggedIn && <Navigation userObj={userObj} />}
        <Routes>
            {isLoggedIn ? (
            <>
                <Route exact path="/" element={<Home userObj={userObj} />} />
                <Route exact path="/profile" element={<Profile userObj={userObj} />} />
            </>
            ) : (
              <Route exact path="/" element={<Auth />} />
            )}
            <Route path="*" element={<span>Not Found</span>} />
        </Routes>
    </Router>
}

export default MyRouter;