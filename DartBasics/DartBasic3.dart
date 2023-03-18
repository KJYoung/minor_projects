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

void main() {
  // --- 3. Functions.  --------------------------------------------
  printFn("Say Hello");
  printFn(stringFormatter("SIU!"));
  printFn(bussinessFormatter(name: 'hi', pos: '33'));
  // printFn(bussinessFormatter2(name: 'hi')); // ERROR
}
