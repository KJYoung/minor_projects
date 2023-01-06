import { dbAddDoc, dbCollection, dbService } from "fbConfig";
import React, { useState } from "react";

const Home = () => {
    const [tweet, setTweet] = useState("");
    return <div>
        <form onSubmit={async (e) =>{
            e.preventDefault();
            try{
                const docRef = await dbAddDoc(dbCollection(dbService, "tweets"), {
                    tweet: tweet,
                    createdAt: Date.now(),
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
    </div>
};
export default Home;