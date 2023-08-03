import 'dart:async';

import 'package:flutter/material.dart';

String formatM(int seconds) {
  var duration = Duration(seconds: seconds);
  var result = duration.toString().split('.')[0].substring(2, 4);
  return result;
}

String formatS(int seconds) {
  var duration = Duration(seconds: seconds);
  var result = duration.toString().split('.')[0].substring(5, 7);
  return result;
}

class ChallDay11 extends StatefulWidget {
  const ChallDay11({
    super.key,
  });

  @override
  State<ChallDay11> createState() => _ChallDay11State();
}

class _ChallDay11State extends State<ChallDay11> {
  int selected = 25,
      startTime = 25 * 60,
      currSec = 25 * 60,
      totalDone = 0,
      totalGoal = 0;
  bool isPlaying = false, isVacation = false;
  late Timer timer;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:
          isVacation ? Colors.blue.shade200 : const Color(0xFFE64D3D),
      body: Padding(
        padding: const EdgeInsets.symmetric(vertical: 45, horizontal: 30),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(
              height: 10,
            ),
            const Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Text(
                  'POMOTIMER',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 26,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(
              height: 130,
            ),
            Row(
              children: [
                const SizedBox(
                  width: 20,
                ),
                Container(
                  child: Row(
                    children: [
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.4),
                          borderRadius: BorderRadius.circular(5.0),
                        ),
                        child: const SizedBox(
                          width: 110,
                          height: 160,
                        ),
                      ),
                      Transform.translate(
                        offset: const Offset(-116, 4),
                        child: Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.6),
                            borderRadius: BorderRadius.circular(5.0),
                          ),
                          child: const SizedBox(
                            width: 122,
                            height: 160,
                          ),
                        ),
                      ),
                      Transform.translate(
                        offset: const Offset(-242, 8),
                        child: Container(
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(5.0),
                          ),
                          child: const SizedBox(
                            width: 130,
                            height: 160,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Transform.translate(
                  offset: const Offset(-180, 0),
                  child: Container(
                    child: Row(
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.4),
                            borderRadius: BorderRadius.circular(5.0),
                          ),
                          child: const SizedBox(
                            width: 110,
                            height: 160,
                          ),
                        ),
                        Transform.translate(
                          offset: const Offset(-116, 4),
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.6),
                              borderRadius: BorderRadius.circular(5.0),
                            ),
                            child: const SizedBox(
                              width: 122,
                              height: 160,
                            ),
                          ),
                        ),
                        Transform.translate(
                          offset: const Offset(-242, 8),
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(5.0),
                            ),
                            child: const SizedBox(
                              width: 130,
                              height: 160,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            Transform.translate(
              offset: const Offset(0, -108),
              child: Row(
                children: [
                  const SizedBox(
                    width: 36,
                  ),
                  Text(
                    formatM(currSec),
                    style: const TextStyle(
                      color: Color(0xFFE64D3D),
                      fontSize: 65,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(width: 34),
                  Transform.translate(
                    offset: const Offset(0, -5),
                    child: const Text(
                      ':',
                      style: TextStyle(
                        color: Color(0xFFf68D6D),
                        fontSize: 65,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(width: 36),
                  Text(
                    formatS(currSec),
                    style: const TextStyle(
                      color: Color(0xFFE64D3D),
                      fontSize: 65,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 2;
                        startTime = 2;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 2,
                      selected: selected == 2,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 15;
                        startTime = 15 * 60;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 15,
                      selected: selected == 15,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 20;
                        startTime = 20 * 60;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 20,
                      selected: selected == 20,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 25;
                        startTime = 25 * 60;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 25,
                      selected: selected == 25,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 30;
                        startTime = 30 * 60;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 30,
                      selected: selected == 30,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        selected = 35;
                        startTime = 35 * 60;
                        currSec = startTime;
                        if (isPlaying) {
                          isPlaying = false;
                          timer.cancel();
                        }
                      });
                    },
                    child: TimeIndicator(
                      time: 35,
                      selected: selected == 35,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(
              height: 40,
            ),
            GestureDetector(
              onTap: () {
                if (isVacation) {
                  // NO.
                } else {
                  if (isPlaying) {
                    timer.cancel();
                    setState(() {
                      isPlaying = false;
                    });
                    return;
                  } else {
                    timer = Timer.periodic(const Duration(seconds: 1), (timer) {
                      if (currSec == 1) {
                        // STOP & RESET
                        timer.cancel();

                        setState(() {
                          currSec = 5;
                          isPlaying = false;
                          isVacation = true;

                          timer = Timer.periodic(const Duration(seconds: 1),
                              (timer) {
                            setState(() {
                              currSec--;

                              if (currSec == 0) {
                                timer.cancel();
                                isVacation = false;
                                currSec = startTime;
                              }
                            });
                          });

                          if (totalDone == 3) {
                            totalDone = 0;
                            totalGoal++;
                          } else {
                            totalDone++;
                          }
                        });
                      } else {
                        setState(() {
                          currSec--;
                          isPlaying = true;
                        });
                      }
                    });
                  }
                }
              },
              child: Center(
                child: Icon(
                  isPlaying
                      ? Icons.pause_circle_outline_rounded
                      : Icons.play_circle_outline_rounded,
                  size: 120,
                ),
              ),
            ),
            const SizedBox(
              height: 60,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Column(
                  children: [
                    Text(
                      '$totalDone / 4',
                      style: TextStyle(
                          fontSize: 24,
                          color: Colors.red.shade200,
                          fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(
                      height: 10,
                    ),
                    const Text(
                      'ROUND',
                      style: TextStyle(
                        fontSize: 23,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Text(
                      '$totalGoal / 12',
                      style: TextStyle(
                          fontSize: 24,
                          color: Colors.red.shade200,
                          fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(
                      height: 10,
                    ),
                    const Text(
                      'GOAL',
                      style: TextStyle(
                        fontSize: 23,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}

class TimeIndicator extends StatelessWidget {
  final int time;
  final bool selected;

  const TimeIndicator({
    super.key,
    required this.time,
    required this.selected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 68,
      height: 45,
      margin: const EdgeInsets.symmetric(horizontal: 5),
      decoration: BoxDecoration(
        color: selected ? Colors.white : const Color(0xFFE64D3D),
        border: Border.all(color: Colors.white),
        borderRadius: BorderRadius.circular(8.0),
      ),
      child: Center(
        child: Text(
          time.toString(),
          style: TextStyle(
            color: selected ? const Color(0xFFE64D3D) : Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.w600,
          ),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}
