import http from "http";
import WebSocket from "ws";
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

const registeredSockets = [];

const wss = new WebSocket.Server({ server });
wss.on("connection", (socket) => {
    socket["nickname"] = "익명의 사용자";
    registeredSockets.push(socket);
    socket.on("close", () => console.log("Disconnected to Browser"));
    socket.on("message", (message) => {
        const msg = JSON.parse(message.toString('utf8'));
        switch(msg.type){
            case "message":
                registeredSockets.forEach(soc => soc.send(`${socket.nickname} : ${msg.payload}`));
                break;
            case "nickname":
                socket["nickname"] = msg.payload;
                break;
            default:
                break;
        }
    });
    console.log("Connected to Browser");
});

server.listen(3000, listenLogger);