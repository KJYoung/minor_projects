import http from "http";
import { Server } from "socket.io";
import { instrument } from "@socket.io/admin-ui";
import express from "express";

const app = express();


app.set('view engine', 'pug');
app.set('views', __dirname + '/views');

app.use('/public', express.static(__dirname + '/public'));
app.get('/', (req, res) => res.render("home"));
app.get('/*', (req, res) => res.redirect('/'));
// Listen

const listenLogger = () => console.log(`Listening on http://localhost:3000`);

const httpServer = http.createServer(app);

const io = new Server(httpServer, {
    cors: {
        origin: ["https://admin.socket.io"],
        credentials: true,
    }
});
instrument(io, {
    auth: false
});

const getPublicRooms = () => {
    const sids = io.sockets.adapter.sids;
    const rooms = io.sockets.adapter.rooms;
    const publicRooms = [];
    rooms.forEach((_, key) => {
        if(sids.get(key) === undefined)
            publicRooms.push(key);
    });
    return publicRooms;
};

const roomSize = (roomName) => {
    return io.sockets.adapter.rooms.get(roomName)?.size;
};

io.on("connection", socket => {
    socket["nickname"] = "익명의 사용자";
    socket.emit("room_list", getPublicRooms());
    socket.on("enter_room", (roomName, doneCallback) => {
        socket.join(roomName);
        doneCallback(roomSize(roomName));
        socket.to(roomName).emit("join", socket.nickname, roomSize(roomName));
        io.sockets.emit("room_list", getPublicRooms());
    });
    socket.on("disconnecting", () => {
        socket.rooms.forEach(room => socket.to(room).emit("left", socket.nickname, roomSize(room) - 1));
    });
    socket.on("disconnect", () => {
        io.sockets.emit("room_list", getPublicRooms());
    })
    socket.on("new_message", (roomName, msg, done) => {
        socket.to(roomName).emit("new_message", socket.nickname, msg);
        done();
    });
    socket.on("nickname", nickname => {
        socket["nickname"] = nickname;
    });
});

httpServer.listen(3000, listenLogger);