import Tweet from "components/Tweet";
import TweetFactory from "components/TweetFactory";
import { authService, dbCollection, dbOrderBy, dbQuery, dbService, onAuthChange, rtOnSnapshot } from "fbConfig";
import React, { useEffect, useState } from "react";

const Home = ({ userObj }) => {
    const [tweetList, setTweetList] = useState([]);

    useEffect(() => {
        const q = dbQuery(
            dbCollection(dbService, "tweets"),
            dbOrderBy("createdAt", "desc")
        );
        const unsubscribe = rtOnSnapshot(q, (snapshot) => {
            const tweetArr = snapshot.docs.map((document) => ({
                id: document.id,
                ...document.data(),
            }));
            setTweetList(tweetArr);
        });
        onAuthChange(authService, (user) => {
            if(user === null)
                unsubscribe();
        }, []);
    }, []);

    return <div className="container">
        <TweetFactory userObj={userObj}/>
        <div style={{ marginTop: 30 }}>
            {tweetList.map(tweet => <Tweet tweet={tweet} key={tweet.id} isAuthor={tweet.author_uid === userObj.uid} />)}
        </div>
    </div>
};
export default Home;