const socket = io();

const welcomeDiv = document.getElementById("welcome");
const welcomeForm = welcomeDiv.querySelector("form");
const roomDiv = document.getElementById("room");
const roomListWrapper = document.getElementById("roomListWrapper");

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
    socket.emit("enter_room", input.value, (roomCount) => {
        welcomeDiv.hidden = true;
        roomDiv.hidden = false;

        const roomTitle = roomDiv.querySelector("h3");
        roomTitle.innerText = `Room ${roomName} : ${roomCount} people`;

        const roomForm = document.getElementById("chatForm");
        
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


socket.on("join", (nickname, roomCount) => {
    const roomTitle = roomDiv.querySelector("h3");
    roomTitle.innerText = `Room ${roomName} : ${roomCount} people`;
    addMessage(`새로운 사람(${nickname})이 입장했습니다!`);
});
socket.on("left", (nickname, roomCount) => {
    const roomTitle = roomDiv.querySelector("h3");
    roomTitle.innerText = `Room ${roomName} : ${roomCount} people`;
    addMessage(`사용자(${nickname})가 퇴장했습니다.`);
});
socket.on("new_message", (nickname, msg) => {
    addMessage(`${nickname} : ${msg}`);
});
socket.on("room_list", (room_list) => {
    const ul = document.createElement("ul");
    room_list.forEach(room => {
        const li = document.createElement("li");
        li.innerText = room;
        ul.appendChild(li);
    });
    roomListWrapper.innerHTML = '';
    roomListWrapper.append(ul);
});