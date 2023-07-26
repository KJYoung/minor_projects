## React Native 기본 공부

- 2023년 7월경.   
- weatherV: openweathermap API로 사용자 지역의 5일간 기상 예보를 보여주는 간단한 JS 기반 앱.
- todoV: TS 기반 앱.

### React Native...
- Native <=> Bridge <=> JavaScript의 Communication!
  - Native에서 발생한/감지된 Event는 JSON message 형태로 JavaScript에 전달되어 로직을 수행하고, response가 다시 JSON message 형태로 Native에 전달되는 형태로 진행된다.
- [개발자 페이지](https://reactnative.dev/) 에서 React Native가 제공하는 Components들을 확인할 수 있다. 예전에는 지금보다 더 많은 components를 제공했었음. 유지관리와 경량화를 위해 필요한 APIs, Components만을 남기는 방향으로 발전한 것. 대신에 Third-party Package(Community Package)를 통해서 기능을 제공하도록 유도함.
- [Third-Party Packages](https://reactnative.directory/) 에서 Third-Party Packages에 대한 정보를 둘러볼 수 있다.
- [Expo SDK](https://docs.expo.dev/versions/latest/) 에서 Expo 팀에서 제작한 Packages, APIs를 볼 수 있다. 커뮤니티에 의존하는 것보다는 안정적일 것.
- Button과 관련된 컴포넌트: TouchableOpacity, TouchableHighlight, TouchableWithoutFeedback, Pressable.
### Minor-Minor Projects
- weatherV : weather application.
### Setting & Debugging
- Expo Init
 ```
    npx create-expo-app --template
 ```
- Expo를 이용한 간단한 테스트.
  ```
    sudo npm install --global expo-cli
    [If MAC user] brew update [takes long time...]
    [If MAC user] brew install watchman
    expo init <Project Name>
  ```