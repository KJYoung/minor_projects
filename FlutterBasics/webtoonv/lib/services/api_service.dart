import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:webtoonv/models/webtoon.dart';

class ApiService {
  static const String baseUrl =
      'https://webtoon-crawler.nomadcoders.workers.dev';
  static const String today = "today";

  static Future<List<WebtoonModel>> getTodaysToons() async {
    List<WebtoonModel> webtoonList = [];

    final response = await http.get(Uri.parse('$baseUrl/$today'));

    if (response.statusCode == 200) {
      final List<dynamic> webtoons = jsonDecode(response.body);
      for (var webtoon in webtoons) {
        webtoonList.add(WebtoonModel.fronJson(webtoon));
      }
      return webtoonList;
    } else {
      return webtoonList;
    }
  }

  static Future<WebtoonDetailModel> getToonDetail(String id) async {
    final response = await http.get(Uri.parse('$baseUrl/$id'));

    if (response.statusCode == 200) {
      final dynamic webtoons = jsonDecode(response.body);
      return WebtoonDetailModel.fronJson(webtoons);
    } else {
      throw Error();
    }
  }

  static Future<List<WebtoonEpisodeModel>> getToonEpisodes(String id) async {
    List<WebtoonEpisodeModel> episodesList = [];
    final response = await http.get(Uri.parse('$baseUrl/$id/episodes'));

    if (response.statusCode == 200) {
      final dynamic episodes = jsonDecode(response.body);
      for (var episode in episodes) {
        episodesList.add(WebtoonEpisodeModel.fronJson(episode));
      }
      return episodesList;
    } else {
      throw Error();
    }
  }
}
