// ignore_for_file: avoid_print

import 'package:challengev/day11.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const App());
}

var whiteStyle = const TextStyle(
  fontWeight: FontWeight.w600,
  color: Colors.white,
);

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: ChallDay11(),
    );
  }
}
