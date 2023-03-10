void main() {
  // --- Variable.  --------------------------------------------
  var variable1 = "Hey! ";
  String variable2 = "Hello World!";
  print("Hello! " + variable1 + variable2);

  var variable3; // dynamic type.
  dynamic variable4; // dynamic type.
  if (variable4 is String) {
    print("Variable4 is String and its length is... " +
        variable4.length.toString());
  }
  ;
  // -- 1.3 About Nullable Variables.
  bool isEmpty(String string) => string.length == 0;
  // isEmpty(null); // Null can't be assigned to String type var.

  String? variableCanbeNull = 'Str';
  variableCanbeNull = null;

  // variableCanbeNull.length; // ERROR! Could be null.
  if (variableCanbeNull != null) {
    variableCanbeNull.length; // Is not error. Null Checked.
  }
  ;
  variableCanbeNull?.length; // OR Use '?'.

  // -- 1.4 Final Variables.
  final String varFin = 'varFin';
  final varFin2 = "varFin2";
  print("Final " + varFin + varFin2);
}
