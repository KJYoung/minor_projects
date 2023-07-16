## React Native 기본 공부

- 2023년 7월경.   

### React Native...
- Native <=> Bridge <=> JavaScript의 Communication!
  - Native에서 발생한/감지된 Event는 JSON message 형태로 JavaScript에 전달되어 로직을 수행하고, response가 다시 JSON message 형태로 Native에 전달되는 형태로 진행된다.

### Minor-Minor Projects
- weatherV : weather application.
### Debugging
- Expo를 이용한 간단한 테스트.
  ```
    sudo npm install --global expo-cli
    [If MAC user] brew update [takes long time...]
    [If MAC user] brew install watchman
    expo init <Project Name>
  ```