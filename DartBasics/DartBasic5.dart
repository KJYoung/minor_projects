// ignore_for_file: unused_local_variable

class Human {
  final String gender;
  Human(this.gender);
  void sayGender() {
    print("I am $gender");
  }
}

class Student extends Human {
  final String name;
  Student(this.name, super.gender);
}

void main() {
  // --- 5. Classes. (2) --------------------------------------------
}
