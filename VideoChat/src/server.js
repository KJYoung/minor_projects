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

const wss = new WebSocket.Server({ server });
wss.on("connection", (socket) => {
    socket.send("hello from Backend");
    socket.on("close", () => console.log("Disconnected to Browser"));
    socket.on("message", (message) => console.log(message.toString('utf8')));
    console.log("Connected to Browser");
});

server.listen(3000, listenLogger);