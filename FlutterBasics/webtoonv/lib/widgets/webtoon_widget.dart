import 'package:flutter/material.dart';
import 'package:webtoonv/models/webtoon.dart';
import 'package:webtoonv/screens/detail_screen.dart';

class WebtoonWidget extends StatelessWidget {
  final WebtoonModel targetWebtoon;

  const WebtoonWidget({super.key, required this.targetWebtoon});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.push(context, MaterialPageRoute(
          builder: (context) {
            return DetailScreen(targetWebtoon: targetWebtoon);
          },
          // fullscreenDialog: true,
        ));
      },
      child: Column(
        children: [
          Hero(
            tag: targetWebtoon.id,
            child: Container(
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      blurRadius: 10,
                      offset: const Offset(0, 8),
                      color: Colors.black.withOpacity(0.3),
                    ),
                  ]),
              height: 300,
              child: Image.network(
                targetWebtoon.thumb,
                headers: const {
                  "User-Agent":
                      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
                },
              ),
            ),
          ),
          const SizedBox(
            height: 22,
          ),
          Text(
            targetWebtoon.title,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
