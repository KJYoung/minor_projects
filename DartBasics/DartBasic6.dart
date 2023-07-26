int fibonacci(int n) {
  if (n == 0 || n == 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

enum PlanetType { terrestrail, gas, ice }

String sayHello(String name, int age, [String? country = 'tel aviv']) {
  return '';
}

String sayHello2(String name, int age, [String? country]) {
  return '';
}

/* Long Comment */
void main() {
  print("Hello, World!"); // Comment.
  print(fibonacci(10));
  var name = 'KJYoung';
  if (name == 'KJYoung') {
    print("String Comparison");
  } else {}
  ;

  int plus(int a, int b, [int? c = 1]) => a + b + (c ?? 0);
  print(plus(1, 2));
  print(plus(1, 2, 44));
}
