import { dbAddDoc, dbCollection, dbGetDocs, dbOrderBy, dbQuery, dbService, rtOnSnapshot } from "fbConfig";
import React, { useEffect, useState } from "react";

const Home = ({ userObj }) => {
    const [tweet, setTweet] = useState("");
    const [tweetList, setTweetList] = useState([]);
    const getTweets = async () => {
        const q = dbQuery(dbCollection(dbService, "tweets"));
        const qSnapshot = await dbGetDocs(q);
        qSnapshot.forEach(doc => {
            const tweetObj = {
                ...doc.data(),
                id: doc.id,
            };
            setTweetList((prev) => [tweetObj, ...prev]);
        });
    }
    useEffect(() => {
        const q = dbQuery(
            dbCollection(dbService, "tweets"),
            dbOrderBy("createdAt", "desc")
        );
        rtOnSnapshot(q, (snapshot) => {
            const tweetArr = snapshot.docs.map((document) => ({
                id: document.id,
                ...document.data(),
            }));
            setTweetList(tweetArr);
        });
    }, []);
    
    return <div>
        <form onSubmit={async (e) =>{
            e.preventDefault();
            try{
                const docRef = await dbAddDoc(dbCollection(dbService, "tweets"), {
                    text: tweet,
                    createdAt: Date.now(),
                    author_uid: userObj.uid,
                });
                setTweet("");
            } catch(error) {
                console.log(error);
            }
        }}>
            <input type="text" placeholder="Type your thought!" maxLength={100}
                   value={tweet} onChange={(e) => setTweet(e.target.value)}/>
            <input type="submit" value="create" />
        </form>
        <div>
            {tweetList.map(tweet => <div key={tweet.id}>
                    <span>{tweet.text}</span>
                </div>)}
        </div>
    </div>
};
export default Home;