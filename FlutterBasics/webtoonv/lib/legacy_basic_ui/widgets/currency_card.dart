import 'package:flutter/material.dart';

class CurrencyCard extends StatelessWidget {
  final String currencyName, amount, currencyCode;
  final IconData icon;
  final bool inverted;

  final Color _black = const Color(0xFF1F2123);

  const CurrencyCard({
    super.key,
    required this.currencyName,
    required this.amount,
    required this.currencyCode,
    required this.icon,
    required this.inverted,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: inverted ? Colors.white : _black,
        borderRadius: BorderRadius.circular(20.0),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(
          vertical: 16,
          horizontal: 20,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  currencyName,
                  style: TextStyle(
                    color: !inverted ? Colors.white : _black,
                    fontSize: 32,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(
                  height: 15,
                ),
                Row(
                  children: [
                    Text(
                      amount,
                      style: TextStyle(
                        fontSize: 21,
                        color: !inverted ? Colors.white : _black,
                      ),
                    ),
                    const SizedBox(
                      width: 7,
                    ),
                    Text(
                      currencyCode,
                      style: TextStyle(
                        fontSize: 18,
                        color: !inverted ? Colors.white : _black,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            Transform.translate(
              offset: const Offset(25, 20),
              child: Transform.scale(
                scale: 1.6,
                child: Icon(
                  icon,
                  color: !inverted ? Colors.white : _black,
                  size: 110,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
