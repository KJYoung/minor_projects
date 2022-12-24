const socket = new WebSocket(`ws://${window.location.host}`);

socket.addEventListener("open", () => {
    console.log("CONNECTED to Server!");
});

socket.addEventListener("message", (message) => {
    console.log("Just got this: ", message.data);
});

socket.addEventListener("close", () => {
    console.log("DISCONNECTED to Server!");
});

setTimeout(() => {
    socket.send("Hello from Browser!");
}, 5000);