const {app, BrowserWindow} = require('electron');
const path = require('path');
const url = require('url');
const { spawn } = require('child_process');

const startDjangoServer = () =>
{
  const djangoBackend = spawn(`/Applications/anaconda3/bin/python`, ['/Users/junyoungkim/workspace/minor_projects/LivingSolo/backend/manage.py', 'runserver', '--noreload']);
  djangoBackend.stdout.on('data', data =>
  {
    console.log(`stdout:\n${data}`);
  });
  djangoBackend.stderr.on('data', data =>
  {
    console.log(`stderr: ${data}`);
  });
  djangoBackend.on('error', (error) =>
  {
    console.log(`error: ${error.message}`);
  });
  djangoBackend.on('close', (code) =>
  {
    console.log(`child process exited with code ${code}`);
  });
  djangoBackend.on('message', (message) =>
  {
    console.log(`message:\n${message}`);
  });
  return djangoBackend;
}

function createWindow() {
    startDjangoServer();

    /*
    * 넓이 1920에 높이 1080의 FHD 풀스크린 앱을 실행시킵니다.
    * */
    const win = new BrowserWindow({
        width:1920,
        height:1080
    });

    /*
    * ELECTRON_START_URL을 직접 제공할경우 해당 URL을 로드합니다.
    * 만일 URL을 따로 지정하지 않을경우 (프로덕션빌드) React 앱이
    * 빌드되는 build 폴더의 index.html 파일을 로드합니다.
    * */
    const startUrl = process.env.ELECTRON_START_URL || url.format({
        pathname: path.join(__dirname, '/../build/index.html'),
        protocol: 'file:',
        slashes: true
    });

    /*
    * startUrl에 배정되는 url을 맨 위에서 생성한 BrowserWindow에서 실행시킵니다.
    * */
    win.loadURL(startUrl);

}

app.on('ready', createWindow);