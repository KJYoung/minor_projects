import 'dart:async';

import 'package:flutter/material.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

const timeAmount = 10;

String format(int seconds) {
  var duration = Duration(seconds: seconds);
  var result = duration.toString().split('.')[0].substring(2);
  return result;
}

class _HomeScreenState extends State<HomeScreen> {
  int totalSeconds = timeAmount, totalDone = 0;
  bool isRunning = false;
  late Timer timer;

  void onStartPressed() {
    if (isRunning) {
      timer.cancel();
      setState(() {
        isRunning = false;
      });
      return;
    } else {
      timer = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (totalSeconds == 1) {
          // STOP & RESET
          timer.cancel();
          setState(() {
            totalSeconds = timeAmount;
            isRunning = false;
            totalDone++;
          });
        } else {
          setState(() {
            totalSeconds--;
            isRunning = true;
          });
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      body: Column(
        children: [
          Flexible(
            flex: 1,
            child: Container(
              alignment: Alignment.center,
              child: Text(
                format(totalSeconds),
                style: TextStyle(
                  color: Theme.of(context).cardColor,
                  fontSize: 75,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          Flexible(
            flex: 2,
            child: Center(
              child: IconButton(
                onPressed: onStartPressed,
                icon: Icon(
                  !isRunning
                      ? Icons.play_circle_outline_rounded
                      : Icons.pause_circle_outline_rounded,
                ),
                iconSize: 150,
                color: Theme.of(context).cardColor,
              ),
            ),
          ),
          Flexible(
            flex: 1,
            child: Row(
              children: [
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardColor,
                      borderRadius: BorderRadius.circular(30.0),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          'PomodoroV',
                          style: TextStyle(
                            fontSize: 25,
                            fontWeight: FontWeight.w600,
                            color:
                                Theme.of(context).textTheme.displayLarge?.color,
                          ),
                        ),
                        Text(
                          '$totalDone',
                          style: TextStyle(
                            fontSize: 40,
                            fontWeight: FontWeight.w200,
                            color:
                                Theme.of(context).textTheme.displayLarge?.color,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
