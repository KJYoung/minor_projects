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

void main() {
  // --- 4. Classes.  --------------------------------------------
  var p1 = Player('VKJY');
  p1.printIntro();
}
