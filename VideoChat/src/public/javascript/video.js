let myStream; 
/** @type {RTCPeerConnection} */
let myPeerConnection;
let muted = true, videoOff = true;

const socket = io();

const myFace = document.getElementById("myFace");
const myAudio = document.getElementById("myAudio");
const myVideo = document.getElementById("myVideo");
const videos  = document.getElementById("videos");

const welcomeDiv = document.getElementById("welcome");
const welcomeForm = welcomeDiv.querySelector("form");
const roomWrapperDiv = document.getElementById("roomWrapper");

let roomName = "";

roomWrapperDiv.hidden = true;

welcomeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = welcomeForm.querySelector("input");
    await startMedia();
    socket.emit("video_join_room", input.value);
    roomName = input.value;
    input.value = "";
});

// Host
socket.on("video_welcome", async () => {
    const offer = await myPeerConnection.createOffer();
    myPeerConnection.setLocalDescription(offer);
    socket.emit("video_offer", roomName, offer);
});
socket.on("video_answer", (answer) => {
    myPeerConnection.setRemoteDescription(answer);
});
// Guest
socket.on("video_offer", async (offer) => {
    myPeerConnection.setRemoteDescription(offer);
    const answer = await myPeerConnection.createAnswer();
    myPeerConnection.setLocalDescription(answer);
    socket.emit("video_answer", roomName, answer);
});
// Both
socket.on("video_ice", (ice) => {
    myPeerConnection.addIceCandidate(ice);
});

myAudio.addEventListener("click", () => {
    if(muted){
        myAudio.innerText = 'Mute';
    }else{
        myAudio.innerText = 'Unmute';
    }
    myStream.getAudioTracks().forEach(track => track.enabled = !track.enabled);
    muted = !muted;
});
myVideo.addEventListener("click", () => {
    if(videoOff){
        myVideo.innerText = "Video Off";
    }else{
        myVideo.innerText = "Video On";
    }
    myStream.getVideoTracks().forEach(track => track.enabled = !track.enabled);
    videoOff = !videoOff;
});
videos.addEventListener("input", async () => {
    await getMedia(videos.value);
    if(myPeerConnection){
        const videoSender = myPeerConnection.getSenders().find(sender => sender.track.kind === "video");
        const videoTrack = myStream.getVideoTracks()[0];
        videoSender.replaceTrack(videoTrack);
    }
});

const startMedia = async () => {
    welcomeDiv.hidden = true;
    roomWrapperDiv.hidden = false;
    await getMedia();
    makeConnection();
};

const getVideoDevices = async () => {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(dev => dev.kind == "videoinput");
        const currentVideoDevices = myStream.getVideoTracks()[0];

        videos.innerHTML = ''; // Reset options
        videoDevices.forEach(dev => {
            const newOption = document.createElement("option");
            newOption.value = dev.deviceId;
            newOption.innerText = dev.label;

            if(currentVideoDevices.label == dev.label)
                newOption.selected = true;
            
            videos.appendChild(newOption);
        })
    } catch(e) {
        console.log(e);
    }
}

const getMedia = async (videoDevId) => {
    if(muted && videoOff){
        // myStream = undefined;
        // myFace.srcObject = myStream;
        // await getVideoDevices();
        // return;
    }
    const constraints = {
        audio: true,
        video: (videoDevId) ? { deviceId: { exact : videoDevId } } : { facingMode: "user" },
    };
    try {
        myStream = await navigator.mediaDevices.getUserMedia(constraints);
        myStream.getAudioTracks().forEach(track => track.enabled = false);
        myStream.getVideoTracks().forEach(track => track.enabled = false);
        myFace.srcObject = myStream;
        await getVideoDevices();
    } catch (e) {
        console.log(e);
    }
};

// WebRTC
const makeConnection = () => {
    myPeerConnection = new RTCPeerConnection({
        iceServers: [
            {
                urls: [
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302",
                    "stun:stun3.l.google.com:19302",
                    "stun:stun4.l.google.com:19302", // STUN server for test
                ],
            },
        ],
    }); // Create a peer connection.
    myPeerConnection.addEventListener("icecandidate", (data) => {
        socket.emit("video_ice", roomName, data.candidate);
    });
    myPeerConnection.addEventListener("addstream", (data) => {
        const peerVideoWrapper = document.getElementById("peerFaceWrapper");
        const peerVideo = peerVideoWrapper.querySelector("video");
        peerVideo.srcObject = data.stream;
    });
    myStream.getTracks().forEach(track => myPeerConnection.addTrack(track, myStream));
}