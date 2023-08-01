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
      home: ChallDay9(),
    );
  }
}

class ChallDay9 extends StatelessWidget {
  const ChallDay9({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
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
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    width: 50,
                    height: 50,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      color: Colors.white,
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: Image.asset('images/test.jpg'),
                  ),
                  const Text(
                    '+',
                    style: TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  )
                ],
              ),
              const SizedBox(
                height: 20,
              ),
              const Row(
                children: [
                  Text('MONDAY 16',
                      style: TextStyle(
                        color: Colors.white,
                      )),
                ],
              ),
              const SizedBox(
                height: 10,
              ),
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'TODAY',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 34,
                      ),
                    ),
                    Text(
                      '•',
                      style: TextStyle(
                        color: Colors.pink.shade700,
                        fontSize: 38,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    Text(
                      ' 17 ',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 37,
                      ),
                    ),
                    Text(
                      ' 18 ',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 37,
                      ),
                    ),
                    Text(
                      ' 19 ',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 37,
                      ),
                    ),
                    Text(
                      ' 20 ',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 37,
                      ),
                    ),
                    Text(
                      ' 21 ',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.4),
                        fontSize: 37,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(
                height: 20,
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(vertical: 18, horizontal: 16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(40),
                  color: Colors.yellow.shade400,
                ),
                child: Column(
                  children: [
                    Row(children: [
                      const Column(
                        children: [
                          Text(
                            '11',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '30',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                          Text(
                            '|',
                            style: TextStyle(
                              fontSize: 20,
                            ),
                          ),
                          Text(
                            '12',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '20',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(
                        width: 20,
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Transform.translate(
                            offset: const Offset(6.5, 3),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'DESIGN',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                          Transform.translate(
                            offset: const Offset(10, -8),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'MEETING',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                        ],
                      )
                    ]),
                    const SizedBox(
                      height: 10,
                    ),
                    Row(
                      children: [
                        const SizedBox(
                          width: 35,
                        ),
                        Text(
                          'ALEX     ',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                        Text(
                          'HELENA     ',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                        Text(
                          'NANA',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                      ],
                    )
                  ],
                ),
              ),
              const SizedBox(
                height: 12,
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(vertical: 18, horizontal: 16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(40),
                  color: Colors.purple.shade200,
                ),
                child: Column(
                  children: [
                    Row(children: [
                      const Column(
                        children: [
                          Text(
                            '12',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '35',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                          Text(
                            '|',
                            style: TextStyle(
                              fontSize: 20,
                            ),
                          ),
                          Text(
                            '14',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '10',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(
                        width: 20,
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Transform.translate(
                            offset: const Offset(1.8, 3),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'DAILY',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                          Transform.translate(
                            offset: const Offset(10, -8),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'PROJECT',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                        ],
                      )
                    ]),
                    const SizedBox(
                      height: 10,
                    ),
                    Row(
                      children: [
                        const SizedBox(
                          width: 35,
                        ),
                        const Text(
                          'ME         ',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: Colors.black,
                          ),
                        ),
                        Text(
                          'RICHARD          CIRY         +4',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                      ],
                    )
                  ],
                ),
              ),
              const SizedBox(
                height: 12,
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(vertical: 18, horizontal: 16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(40),
                  color: Colors.green.shade300,
                ),
                child: Column(
                  children: [
                    Row(children: [
                      const Column(
                        children: [
                          Text(
                            '15',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '00',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                          Text(
                            '|',
                            style: TextStyle(
                              fontSize: 20,
                            ),
                          ),
                          Text(
                            '16',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '30',
                            style: TextStyle(
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(
                        width: 20,
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Transform.translate(
                            offset: const Offset(6.5, 3),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'WEEKLY',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                          Transform.translate(
                            offset: const Offset(10, -8),
                            child: Transform.scale(
                              scale: 1.2,
                              child: const Text(
                                'PLANNING',
                                style: TextStyle(
                                  fontSize: 47,
                                  fontWeight: FontWeight.w400,
                                ),
                              ),
                            ),
                          ),
                        ],
                      )
                    ]),
                    const SizedBox(
                      height: 10,
                    ),
                    Row(
                      children: [
                        const SizedBox(
                          width: 35,
                        ),
                        Text(
                          'DEN     ',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                        Text(
                          'NANA     ',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                        Text(
                          'MARK',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: Colors.black.withOpacity(0.5),
                          ),
                        ),
                      ],
                    )
                  ],
                ),
              )
            ],
          ),
        ));
  }
}
