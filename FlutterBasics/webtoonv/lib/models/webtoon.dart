class WebtoonModel {
  final String id, title, thumb;

  WebtoonModel({required this.id, required this.title, required this.thumb});

// Named Constructor
  WebtoonModel.fronJson(Map<String, dynamic> json)
      : title = json['title'],
        thumb = json['thumb'],
        id = json['id'];
}

class WebtoonDetailModel {
  final String title, about, genre, age;

// Named Constructor
  WebtoonDetailModel.fronJson(Map<String, dynamic> json)
      : title = json['title'],
        about = json['about'],
        genre = json['genre'],
        age = json['age'];
}

class WebtoonEpisodeModel {
  final String id, thumb, title, rating, date;

// Named Constructor
  WebtoonEpisodeModel.fronJson(Map<String, dynamic> json)
      : id = json['id'],
        title = json['title'],
        thumb = json['thumb'],
        rating = json['rating'],
        date = json['date'];
}
