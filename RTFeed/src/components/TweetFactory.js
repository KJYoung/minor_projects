import { faPlus, faTimes } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { dbAddDoc, dbCollection, dbService, storageGetDownloadURL, storageRef, storageService, storageUploadString } from "fbConfig";
import React, { useRef, useState } from "react";
import { v4 } from "uuid";

const TweetFactory = ({ userObj }) => {
    const [tweet, setTweet] = useState("");
    const [imagePreview, setImagePreview] = useState(null);
    const imageInput = useRef();

    const onImageChange = (event) => {
        const {target : {files}} = event; // files : file list
        if(files.length > 1){
            window.alert("한 개의 사진만 선택하세요.");
            return;
        }else if(files.length === 0){
            return; // No image selected.
        }
        // Here : One Image File is Successfully Selected.
        const image = files[0];
        const fileReader = new FileReader();
        fileReader.onloadend = (fEvent) => {
            setImagePreview(fEvent.currentTarget.result);
        };
        fileReader.readAsDataURL(image);
    };

    return <form onSubmit={async (e) =>{
        e.preventDefault();

        if(tweet === ""){
            return;
        }
        let imgURL = "";
        // 1. Upload the photo if any.
        if(imagePreview !== null){
            const fileRef = storageRef(storageService, `${userObj.uid}/${v4()}`);
            const response = await storageUploadString(fileRef, imagePreview, "data_url");
            imgURL = await storageGetDownloadURL(response.ref);
        }
        // 2. Create a new tweet.
        try{
            await dbAddDoc(dbCollection(dbService, "tweets"), {
                text: tweet,
                createdAt: Date.now(),
                author_uid: userObj.uid,
                img_url: imgURL, // "" if there was no photo.
            });
            setTweet("");
            setImagePreview(null);
            imageInput.current.value = "";
        } catch(error) {
            console.log(error);
        }
    }} className="factoryForm">
        <div className="factoryInput__container">
            <input type="text" placeholder="Type your thought!" maxLength={100}
                value={tweet} onChange={(e) => setTweet(e.target.value)}
                className="factoryInput__input" />
            <input type="submit" value="&rarr;" className="factoryInput__arrow" />
        </div>
        <label htmlFor="attach-file" className="factoryInput__label">
            <span>Add photos</span>
            <FontAwesomeIcon icon={faPlus} />
        </label>
        <input id="attach-file" type="file" accept="image/*" ref={imageInput} onChange={onImageChange} style={{ opacity: 0 }}/>
        {imagePreview && <div className="factoryForm__attachment" >
            <img src={imagePreview} alt="preview" style={{ backgroundImage: imagePreview }}/>
            <div onClick={() => {
                setImagePreview(null);
                imageInput.current.value = "";
            }} className="factoryForm__clear">
                <span>Clear</span>
                <FontAwesomeIcon icon={faTimes} />
            </div>
        </div>}
        
    </form>
};

export default TweetFactory;