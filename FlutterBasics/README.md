## Flutter 기본 공부

- 2023년 7월경.   

### Flutter...
- Flutter는 게임 엔진들의 작용방식처럼 엔진이 구동하고, 우리의 Flutter(Dart) Code를 그 엔진이 실행하는 방식으로 작동한다. 그렇기 때문에, Host OS의 Native Widget을 이용하는게 아니라 Flutter 자체적으로 렌더링을 진행하게 되고, 모든 것은 허상인 셈이다. 이는 오히려 Host OS의 통제에서 자유로워진다는 장점으로도 작용한다. 아무튼 Host OS와 상호작용할 일은 없다.
- 그렇기 때문에 용도에 따라 React Native와 Flutter를 다르게 사용할 수 있으므로, React Native, Flutter 모두를 배워두는 것은 나쁘지 않을 것! 당연히 Native Widget들이 많이 필요하면 React Native를 사용하는 것이 편할 것이기 때문이다.
- Declarative UI Programming을 조금씩 해보니 별로 유쾌한 경험은 아니었다. 새로운(신선한) 경험이긴 했지만. 물론, VSCode에서의 개발 경험 자체는 정말 우수하다. Formatting도 정말 깔끔하고.
  - User Setting.json(Command Pallete > Open User Settings(JSON))에 `"dart.previewFlutterUiGuides": true` 옵션을 주면 UI Hierarchy를 깔끔하게 보여준다! 설정을 활성화시킨 후에 VSCode를 재실행해줘야 작동하기 시작한다.
  - Widget Hierarchy에서 각 Widget에 커서를 두고 `command+.`을 누르거나, 왼쪽에 뜨는 전구 버튼을 누르면 Code Actions 이용가능. (ex) Wrap with Padding과 같이 매우 유용하다.
  - 이런 기능들을 보다보니 코드 가독성은 좀 떨어져도 Developer Tools들로 커버가 가능한 느낌?
- TODO: Java, Kotlin


### Debugging