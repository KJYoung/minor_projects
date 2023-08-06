import 'package:flutter/material.dart';
import 'package:challengev/models/movie.dart';
import 'package:challengev/services/api_service.dart';
import 'package:challengev/widgets/webtoon_widget.dart';

class HomeScreen extends StatelessWidget {
  HomeScreen({super.key});

  final Future<List<List<MovieModel>>> movies = ApiService.getMovies();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text(
          "Movies",
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.w500,
          ),
        ),
        backgroundColor: Colors.black,
        foregroundColor: Colors.amber,
        elevation: 3,
      ),
      body: FutureBuilder(
        future: movies,
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 20,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(
                    height: 50,
                  ),
                  const Text(
                    'Popular Movies',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  Expanded(child: makeList(snapshot.data![0])),
                  const SizedBox(
                    height: 10,
                  ),
                  const Text(
                    'Now in Cinemas',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  Expanded(child: makeList(snapshot.data![1])),
                  const SizedBox(
                    height: 10,
                  ),
                  const Text(
                    'Coming Soon',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  Expanded(child: makeList(snapshot.data![2])),
                ],
              ),
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }

  ListView makeList(List<MovieModel> snapshot) {
    return ListView.separated(
      scrollDirection: Axis.horizontal,
      itemCount: snapshot.length,
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
      itemBuilder: (context, index) {
        var targetMovie = snapshot[index];
        return MovieWidget(targetMovie: targetMovie);
      },
      separatorBuilder: (context, index) {
        return const SizedBox(
          width: 50,
        );
      },
    );
  }
}
