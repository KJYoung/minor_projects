const socket = io();

const welcomeDiv = document.getElementById("welcome");
const welcomeForm = welcomeDiv.querySelector("form");
const roomDiv = document.getElementById("room");

roomDiv.hidden = true;

let roomName = "";

const addMessage = (message) => {
    const ul = roomDiv.querySelector("ul");
    const li = document.createElement("li");
    li.innerText = message;
    ul.appendChild(li);
}

const nickForm = document.getElementById("nickForm");
nickForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = nickForm.querySelector("input");
    socket.emit("nickname", input.value);
});

welcomeForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = welcomeForm.querySelector("input");
    roomName = input.value;

    // event type, payloads, callback - called by server(executed in front-end)
    // callback should be the last argument if any.
    socket.emit("enter_room", input.value, () => {
        welcomeDiv.hidden = true;
        roomDiv.hidden = false;

        const roomTitle = roomDiv.querySelector("h3");
        const roomForm = document.getElementById("chatForm");
        
        roomTitle.innerText = `Room ${roomName}`;
        
        roomForm.addEventListener("submit", (event) => {
            event.preventDefault();
            const input = roomForm.querySelector("input");
            const value = input.value;
            socket.emit("new_message", roomName, value, () => {
                addMessage(`You: ${value}`);
            });
            input.value = "";
        });
    });

    input.value = "";
});



socket.on("join", (nickname) => {
    addMessage(`새로운 사람(${nickname})이 입장했습니다!`);
});
socket.on("left", (nickname) => {
    addMessage(`사용자(${nickname})가 퇴장했습니다.`);
});
socket.on("new_message", (nickname, msg) => {
    addMessage(`${nickname} : ${msg}`);
})