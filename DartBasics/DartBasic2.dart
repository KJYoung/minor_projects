// ignore_for_file: unused_local_variable

void main() {
  // --- 2. Data Types.  --------------------------------------------
  String name = 'VKJY';
  int number = 3;
  double number21 = 3;
  double number22 = 3.4;
  bool boolean = true;
  num generalNumber1 = 3;
  num generalNumber2 = 3.4;

  // Lists.
  var numbers = [1, 2, 3, 4, 5];
  numbers.add(332);
  List<bool> bools = [true, true, false];
  var giveMeThree = true;
  var numberCollectionIF = [1, 2, if (giveMeThree) 3, 4]; // Collection IF.
  print(numberCollectionIF);
}
