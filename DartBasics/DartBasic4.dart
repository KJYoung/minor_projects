// ignore_for_file: unused_local_variable

class Player {
  late String name; // Should be 'LATE' var!

  // Constructor.
  Player(String name) {
    this.name = name;
  }

  void printIntro() {
    print("Hi. I am $name"); // Instead of this.name
  }
}

class PlayerShortCut {
  String name;

  // Shortcut Constructor.
  PlayerShortCut(this.name);
}

class PlayerNamedParams {
  String name;

  // Named Params Constructor.
  PlayerNamedParams({required this.name});
}

class PlayerNamedConst {
  String name;
  late int state;

  // Named Constructor.
  PlayerNamedConst({required this.name}) : this.state = 0;

  PlayerNamedConst.createWithState({required name, required this.state})
      : this.name = name;
}

void main() {
  // --- 4. Classes.  --------------------------------------------
  var p1 = Player('VKJY');
  p1.printIntro();

  var p2 = PlayerNamedConst(name: 'jason');
  var p3 = PlayerNamedConst.createWithState(name: 'juhe', state: 1);
}
