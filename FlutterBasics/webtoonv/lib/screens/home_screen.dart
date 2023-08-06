import 'package:flutter/material.dart';
import 'package:webtoonv/models/webtoon.dart';
import 'package:webtoonv/services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<WebtoonModel> webtoons = [];
  bool isLoading = true;

  void waitForWebtoons() async {
    webtoons = await ApiService.getTodaysToons();
    setState(() {
      isLoading = false;
    });
  }

  @override
  void initState() {
    super.initState();
    waitForWebtoons();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text(
          "오늘의 웹툰",
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.w500,
          ),
        ),
        backgroundColor: Colors.black,
        foregroundColor: Colors.amber,
        elevation: 3,
      ),
      body: Container(),
    );
  }
}
