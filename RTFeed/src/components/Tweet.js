import { faPencilAlt, faTrash } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { dbDeleteDoc, dbDoc, dbService, dbUpdateDoc, storageDeleteObj, storageRef, storageService } from "fbConfig";
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

            // Delete Photo if any
            if(tweet.img_url && tweet.img_url !== ""){
                await storageDeleteObj(storageRef(storageService, tweet.img_url));
            }
        }
    };
    const onEdit = async (event) => {
        event.preventDefault();
        const tweetRef = dbDoc(dbService, "tweets", `${tweet.id}`);
        await dbUpdateDoc(tweetRef, { text: text });
        setIsEditing(false);
    };
    
    return (
        <div className="nweet">
            {isEditing ? <>
                <form onSubmit={onEdit} className="container nweetEdit">
                    <input type="text" value={text} onChange={e => setText(e.target.value)} required autoFocus className="formInput" />
                    <input type="submit" value="confirm" className="formBtn"/>
                </form>
                <span onClick={toggleEdit} className="formBtn cancelBtn">Cancel</span>
            </> : <>
                <h4>{tweet.text}</h4>
                {tweet.img_url && <img src={tweet.img_url} alt="tweetAttachment" />}
                {isAuthor && <div className="nweet__actions">
                    <span onClick={toggleEdit}>
                        <FontAwesomeIcon icon={faPencilAlt} />
                    </span>
                    <span onClick={onDelete}>
                        <FontAwesomeIcon icon={faTrash} />
                    </span>
                </div>}
            </>}
            
        </div>
    );
};

export default Tweet;