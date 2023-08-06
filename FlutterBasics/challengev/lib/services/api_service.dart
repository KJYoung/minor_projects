import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:challengev/models/movie.dart';

class ApiService {
  static const String baseUrl = 'https://movies-api.nomadcoders.workers.dev';
  static const String popular = "popular";
  static const String nowPlaying = "now-playing";
  static const String comingSoon = "coming-soon";
  static const String detail = "mover?id=";

  static Future<List<List<MovieModel>>> getMovies() async {
    List<List<MovieModel>> movieList = [[], [], []];

    final resPop = await http.get(Uri.parse('$baseUrl/$popular'));
    if (resPop.statusCode == 200) {
      List<dynamic> movies = jsonDecode(resPop.body)['results'];

      for (var movie in movies) {
        movieList[0].add(MovieModel.fronJson(movie));
      }
    }
    final resNow = await http.get(Uri.parse('$baseUrl/$nowPlaying'));
    if (resNow.statusCode == 200) {
      final List<dynamic> movies = jsonDecode(resNow.body)['results'];
      for (var movie in movies) {
        movieList[1].add(MovieModel.fronJson(movie));
      }
    }
    final resCom = await http.get(Uri.parse('$baseUrl/$comingSoon'));
    if (resCom.statusCode == 200) {
      final List<dynamic> movies = jsonDecode(resCom.body)['results'];
      for (var movie in movies) {
        movieList[2].add(MovieModel.fronJson(movie));
      }
    }

    return movieList;
  }

  static Future<MovieDetailModel> getToonDetail(String id) async {
    final response = await http.get(Uri.parse('$baseUrl/$id'));

    if (response.statusCode == 200) {
      final dynamic movies = jsonDecode(response.body);
      return MovieDetailModel.fronJson(movies);
    } else {
      throw Error();
    }
  }

  static Future<List<MovieEpisodeModel>> getToonEpisodes(String id) async {
    List<MovieEpisodeModel> episodesList = [];
    final response = await http.get(Uri.parse('$baseUrl/$id/episodes'));

    if (response.statusCode == 200) {
      final dynamic episodes = jsonDecode(response.body);
      for (var episode in episodes) {
        episodesList.add(MovieEpisodeModel.fronJson(episode));
      }
      return episodesList;
    } else {
      throw Error();
    }
  }
}
