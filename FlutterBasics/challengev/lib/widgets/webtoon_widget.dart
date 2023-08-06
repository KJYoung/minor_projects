import 'package:challengev/screens/detail_screen.dart';
import 'package:flutter/material.dart';
import 'package:challengev/models/movie.dart';

class MovieWidget extends StatelessWidget {
  final MovieModel targetMovie;

  const MovieWidget({super.key, required this.targetMovie});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.push(context, MaterialPageRoute(
          builder: (context) {
            return DetailScreen(targetMovie: targetMovie);
          },
        ));
      },
      child: Column(
        children: [
          Container(
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
            height: 100,
            child: Image.network(
              "https://image.tmdb.org/t/p/w500${targetMovie.posterPath}",
              headers: const {
                "User-Agent":
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
              },
            ),
          ),
          const SizedBox(
            height: 22,
          ),
          Text(
            targetMovie.title,
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
