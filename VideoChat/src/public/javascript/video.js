let myStream, muted = true, videoOff = true;

const socket = io();

const myFace = document.getElementById("myFace");
const myAudio = document.getElementById("myAudio");
const myVideo = document.getElementById("myVideo");
const videos  = document.getElementById("videos");

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
});

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



getMedia();