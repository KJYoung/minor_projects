import http from "http";
import SocketIO from "socket.io";
import express from "express";

const app = express();


app.set('view engine', 'pug');
app.set('views', __dirname + '/views');

app.use('/public', express.static(__dirname + '/public'));
app.get('/', (req, res) => res.render("home"));
app.get('/*', (req, res) => res.redirect('/'));
// Listen

const listenLogger = () => console.log(`Listening on http://localhost:3000`);

const server = http.createServer(app);


const io = SocketIO(server);

io.on("connection", socket => {
    socket["nickname"] = "익명의 사용자";
    socket.on("enter_room", (roomName, doneCallback) => {
        socket.join(roomName);
        doneCallback();
        socket.to(roomName).emit("join", socket.nickname);
    });
    socket.on("disconnecting", () => {
        socket.rooms.forEach(room => socket.to(room).emit("left", socket.nickname));
    });
    socket.on("new_message", (roomName, msg, done) => {
        socket.to(roomName).emit("new_message", socket.nickname, msg);
        done();
    });
    socket.on("nickname", nickname => {
        socket["nickname"] = nickname;
    });
});

server.listen(3000, listenLogger);