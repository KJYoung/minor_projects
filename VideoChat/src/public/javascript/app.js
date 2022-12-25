const socket = io();

const welcomeDiv = document.getElementById("welcome");
const welcomeForm = welcomeDiv.querySelector("form");

welcomeForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = welcomeForm.querySelector("input");

    // event type, payloads, callback - called by server(executed in front-end)
    // callback should be the last argument if any.
    socket.emit("enter_room", input.value, (msg) => {
        console.log(msg);
    });

    input.value = "";
});