import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webtoonv/models/webtoon.dart';
import 'package:webtoonv/services/api_service.dart';
import 'package:webtoonv/widgets/episode_widget.dart';

class DetailScreen extends StatefulWidget {
  final WebtoonModel targetWebtoon;

  const DetailScreen({super.key, required this.targetWebtoon});

  @override
  State<DetailScreen> createState() => _DetailScreenState();
}

class _DetailScreenState extends State<DetailScreen> {
  late Future<WebtoonDetailModel> webtoon;
  late Future<List<WebtoonEpisodeModel>> episodes;
  late SharedPreferences prefs;

  bool isLiked = false;

  Future initPref() async {
    prefs = await SharedPreferences.getInstance();
    final likedToons = prefs.getStringList('likedToons');
    if (likedToons != null) {
      if (likedToons.contains(widget.targetWebtoon.id) == true) {
        isLiked = true;
        setState(() {});
      }
    } else {
      await prefs.setStringList('likedToons', []);
    }
  }

  onFavTap() async {
    final likedToons = prefs.getStringList('likedToons');
    if (likedToons != null) {
      if (isLiked) {
        likedToons.remove(widget.targetWebtoon.id);
      } else {
        likedToons.add(widget.targetWebtoon.id);
      }
      await prefs.setStringList('likedToons', likedToons);
      setState(() {
        isLiked = !isLiked;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    webtoon = ApiService.getToonDetail(widget.targetWebtoon.id);
    episodes = ApiService.getToonEpisodes(widget.targetWebtoon.id);

    initPref();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(
          widget.targetWebtoon.title,
          style: const TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.w500,
          ),
        ),
        backgroundColor: Colors.black,
        foregroundColor: Colors.amber,
        elevation: 3,
        actions: [
          IconButton(
            onPressed: onFavTap,
            icon: Icon(
              isLiked ? Icons.favorite_rounded : Icons.favorite_outline_rounded,
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 0, vertical: 50),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Hero(
                  tag: widget.targetWebtoon.id,
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
                      widget.targetWebtoon.thumb,
                      headers: const {
                        "User-Agent":
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
                      },
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(
              height: 30,
            ),
            FutureBuilder(
              future: webtoon,
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 45),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '장르 : ${snapshot.data!.genre} / ${snapshot.data!.age}',
                          style: const TextStyle(
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(
                          height: 10,
                        ),
                        Text(
                          snapshot.data!.about,
                          style: const TextStyle(
                            fontSize: 16,
                          ),
                        ),
                      ],
                    ),
                  );
                } else {
                  return const Text("...");
                }
              },
            ),
            const SizedBox(
              height: 20,
            ),
            FutureBuilder(
              future: episodes,
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 35),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        for (var episode in snapshot.data!)
                          Episode(
                            episode: episode,
                            webtoonId: widget.targetWebtoon.id,
                          ),
                      ],
                    ),
                  );
                } else {
                  return const SizedBox();
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}
