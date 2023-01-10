import { dbDeleteDoc, dbDoc, dbService, dbUpdateDoc } from "fbConfig";
import React, { useState } from "react";

const Tweet = ({tweet, isAuthor}) => {
    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(tweet.text);

    const toggleEdit = () => setIsEditing((prev) => !prev);
    // Delete, Edit Handlers
    const onDelete = async () => {
        const ok = window.confirm("정말로 삭제하시겠습니까?");
        if(ok){
            const tweetRef = dbDoc(dbService, "tweets", `${tweet.id}`);
            await dbDeleteDoc(tweetRef);
        }
    };
    const onEdit = async (event) => {
        event.preventDefault();
        const tweetRef = dbDoc(dbService, "tweets", `${tweet.id}`);
        await dbUpdateDoc(tweetRef, { text: text });
        setIsEditing(false);
    };
    
    return (
        <div>
            {isEditing ? <>
                <form onSubmit={onEdit}>
                    <input type="text" value={text} onChange={e => setText(e.target.value)} required />
                    <input type="submit" value="confirm" />
                </form>
                <button onClick={toggleEdit}>Cancel</button>
            </> : <>
                <span>{tweet.text}</span>
                {tweet.img_url && <img src={tweet.img_url} width="50px" height="50px" alt="tweetAttachment" />}
                {isAuthor && <>
                    <button onClick={toggleEdit}>Edit</button>
                    <button onClick={onDelete}>Delete</button>
                </>}
            </>}
            
        </div>
    );
};

export default Tweet;