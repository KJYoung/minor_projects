const msgList = document.querySelector("ul");
const msgForm = document.querySelector("#msgForm");
const nickForm = document.querySelector("#nickForm");

const socket = new WebSocket(`ws://${window.location.host}`);

const makeMessage = (type, payload) => {
    return JSON.stringify({
        type: type,
        payload: payload
    });
}

socket.addEventListener("open", () => {
    console.log("CONNECTED to Server!");
});

socket.addEventListener("message", (message) => {
    const li = document.createElement("li");
    li.innerText = message.data;
    msgList.append(li);
});

socket.addEventListener("close", () => {
    console.log("DISCONNECTED to Server!");
});

nickForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = nickForm.querySelector("input");
    socket.send(makeMessage("nickname", input.value));
});

msgForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = msgForm.querySelector("input");
    socket.send(makeMessage("message", input.value));
    input.value = "";

    // My Message.
    const li = document.createElement("li");
    li.innerText = `You : ${input.value}`;
    msgList.append(li);
});