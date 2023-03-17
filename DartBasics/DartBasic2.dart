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

  // String Interpolation.
  var interp = 'Hello, $name. ${number + 2}';

  // Collection For.
  var oldList = [1, 2, 3];
  var newList = [4545, 5454, for (var num in oldList) num + 70];
  print(newList);

  // Map
  var OBJ = {'name': 'VKJYOUNG', 'age': 23, 32: 'jj'};
  Map<int, bool> numBool = {
    1: true,
    2: false,
  };

  // Set
  var SET = {1, 2, 3, 4}; // Unique Elements
  print(SET);
  SET.add(4);
  print(SET);
}
