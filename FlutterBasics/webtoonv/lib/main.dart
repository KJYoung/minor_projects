import 'package:flutter/material.dart';
import 'package:webtoonv/screens/home_screen.dart';

void main() async {
  runApp(const App());
}

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  int counter = 0;
  List<int> numbers = [];

  void onClicked() {
    setState(() {
      counter++;
      numbers.add(numbers.length);
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        textTheme: const TextTheme(
          titleLarge: TextStyle(
            color: Colors.red,
          ),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class TitleWidget extends StatefulWidget {
  const TitleWidget({
    super.key,
  });

  @override
  State<TitleWidget> createState() => _TitleWidgetState();
}

class _TitleWidgetState extends State<TitleWidget> {
  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Text('My Large Title',
        style: TextStyle(color: Theme.of(context).textTheme.titleLarge?.color));
  }
}
