import express from "express";

const app = express();

console.log("Project Initialized");

app.set('view engine', 'pug');
app.set('views', __dirname + '/views');

app.use('/public', express.static(__dirname + '/public'));
app.get('/', (req, res) => res.render("home"));
app.get('/*', (req, res) => res.redirect('/'));
// Listen
app.listen(3000, () => console.log(`Listening on http://localhost:3000`));