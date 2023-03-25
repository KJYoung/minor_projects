// ignore_for_file: unused_local_variable

class Strong {
  // Mixin. without Constructor!!!
  final double magnitude = 1.0;
}

class Charm {
  final double level = 1.0;
}

class Intelligence {
  final double level = 1.5;
}

class Human {
  final String gender;
  Human(this.gender);
  void sayGender() {
    print("I am $gender");
  }
}

// class Student extends Human with Strong, Charm, Intelligence {
class Student extends Human with Strong, Intelligence, Charm {
  // with 뒤에 오는 순서에 따라 overload될 수 있다.
  // with Charm, Intelligence 하면 level은 1.5가 되지만,
  // with Intelligence, Charm 하면 level은 1.0이 된다.
  final String name;
  // Student(this.name, super.gender); // Also works!
  Student(this.name, String gender) : super(gender);

  @override
  void sayGender() {
    // override
    print("I am $gender student!");
  }
}

void main() {
  // --- 5. Classes. (2) --------------------------------------------
  var stdt1 = Student('KJY', 'male');
  stdt1.sayGender();
  print(stdt1.level);
}
