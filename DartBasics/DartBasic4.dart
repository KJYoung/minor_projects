// ignore_for_file: unused_local_variable

abstract class Human {
  void breathe();
  void walk();
  void eat();
  void drink();
}

class Player extends Human {
  late String name; // Should be 'LATE' var!

  // Constructor.
  Player(String name) {
    this.name = name;
  }

  void printIntro() {
    print("Hi. I am $name"); // Instead of this.name
  }

  void breathe() {
    print("Whoo..");
  }

  void walk() {
    print("!");
  }

  void eat() {
    print("Yammy");
  }

  void drink() {
    print("Yummy");
  }
}

class PlayerShortCut {
  String name;

  // Shortcut Constructor.
  PlayerShortCut(this.name);
}

class PlayerNamedParams {
  String name;
  int age;
  GRADE grade;

  // Named Params Constructor.
  PlayerNamedParams(
      {required this.name, required this.age, required this.grade});
}

class PlayerNamedConst {
  String name;
  late int state;

  // Named Constructor.
  PlayerNamedConst({required this.name}) : this.state = 0;

  PlayerNamedConst.createWithState({required name, required this.state})
      : this.name = name;
}

enum GRADE { A, B }

void main() {
  // --- 4. Classes.  --------------------------------------------
  var p1 = Player('VKJY');
  p1.printIntro();

  var p2 = PlayerNamedConst(name: 'jason');
  var p3 = PlayerNamedConst.createWithState(name: 'juhe', state: 1);

  var p4 = PlayerNamedParams(name: 'JY', age: 23, grade: GRADE.A)
    // p4.name = 'KJY';
    // p4.age = 24;
    ..name = 'KJY'
    ..age = 22;
  // CASCADE NOTATION!
}
