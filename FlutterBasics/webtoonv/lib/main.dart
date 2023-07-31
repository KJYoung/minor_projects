import 'package:flutter/material.dart';
import 'package:webtoonv/widgets/button.dart';

void main() {
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
          backgroundColor: const Color(0xFF181818),
          body: Padding(
            padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(
                  height: 60,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        const Text(
                          'Hey, Junyoung',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 34,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        Text(
                          'Good Evening..!',
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.6),
                            fontSize: 22,
                          ),
                        ),
                      ],
                    )
                  ],
                ),
                const SizedBox(
                  height: 60,
                ),
                Text('Total Balance',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.5),
                      fontSize: 25,
                    )),
                const SizedBox(
                  height: 10,
                ),
                const Text('\$1,234,567',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 40,
                      fontWeight: FontWeight.w700,
                    )),
                const SizedBox(
                  height: 20,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Button(
                      text: 'Transfer',
                      bgColor: Colors.amber.shade400,
                      txtColor: Colors.black,
                    ),
                    Button(
                      text: 'Request',
                      bgColor: Colors.grey.shade700,
                      txtColor: Colors.white,
                    ),
                  ],
                ),
                const Row(
                  children: [Text('Hi')],
                ),
              ],
            ),
          )),
    );
  }
}
