// ignore_for_file: unused_local_variable

void printFn(String content) {
  print(content);
}

// Direct Return.
String stringFormatter(String content) => "{{{ $content }}}";
// Named Parameter.
String bussinessFormatter({String name = "annonymous", String pos = "void"}) =>
    "Hi, I am $name, $pos";
String bussinessFormatter2({required String name, required String pos}) =>
    "Hi, I am $name, $pos";
// Optional Parameter.
String bussinessFormatter3(String name, String pos, [String? addit]) =>
    "Hi, I am $name, $pos + $addit";
// QQ Operator.
String capitalizeStr(String str) => str.toUpperCase();
String capitalizeStrNULL(String? str) {
  if (str != null) {
    return str.toUpperCase();
  } else {
    return '';
  }
}

String capitalizeStrNULL2(String? str) => str != null ? str.toUpperCase() : '';
String capitalizeStrNULL3(String? str) => str?.toUpperCase() ?? '';

// Typedef
List<int> reverseListINT(List<int> lst) {
  var reversed = lst.reversed;
  return reversed.toList();
}

typedef IntList = List<int>;
IntList reverseListINT2(IntList lst) {
  var reversed = lst.reversed;
  return reversed.toList();
}

void main() {
  // --- 3. Functions.  --------------------------------------------
  printFn("Say Hello");
  printFn(stringFormatter("SIU!"));
  printFn(bussinessFormatter(name: 'hi', pos: '33'));
  // printFn(bussinessFormatter2(name: 'hi')); // ERROR
  printFn(bussinessFormatter3('Messia', '23'));
  printFn(bussinessFormatter3('Messia', '23', 'æ'));
  printFn(capitalizeStr('junyoung kim'));

  String? str1;
  str1 ??= 'Hey'; // Assign if str1 is null.
  print(str1);
  str1 ??=
      'Joseph'; // Assign if str1 is null. i.e. This line would have no effect.
  print(str1);
}
