import 'package:flutter/material.dart';
import 'package:webtoonv/widgets/button.dart';
import 'package:webtoonv/widgets/currency_card.dart';

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
        body: SingleChildScrollView(
          child: Padding(
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
                  height: 20,
                ),
                Text('Total Balance',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.5),
                      fontSize: 25,
                    )),
                const SizedBox(
                  height: 10,
                ),
                const Text(
                  '\$1,234,567',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 40,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(
                  height: 40,
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
                const SizedBox(
                  height: 40,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'Wallets',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 32,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      'View All',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 18,
                      ),
                    ),
                  ],
                ),
                const SizedBox(
                  height: 20,
                ),
                const CurrencyCard(
                  currencyName: 'EURO',
                  currencyCode: 'EUR',
                  amount: '6 576',
                  icon: Icons.euro_rounded,
                  inverted: false,
                ),
                Transform.translate(
                  offset: const Offset(0, -30),
                  child: const CurrencyCard(
                    currencyName: 'Bitcoin',
                    currencyCode: 'BTC',
                    amount: '9 123',
                    icon: Icons.currency_bitcoin_rounded,
                    inverted: true,
                  ),
                ),
                Transform.translate(
                  offset: const Offset(0, -60),
                  child: const CurrencyCard(
                    currencyName: 'Dollar',
                    currencyCode: 'USD',
                    amount: '1 023',
                    icon: Icons.attach_money_rounded,
                    inverted: false,
                  ),
                ),
                Transform.translate(
                  offset: const Offset(0, -90),
                  child: const CurrencyCard(
                    currencyName: 'Dollar',
                    currencyCode: 'USD',
                    amount: '1 023',
                    icon: Icons.attach_money_rounded,
                    inverted: true,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
