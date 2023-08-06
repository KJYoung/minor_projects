class WebtoonModel {
  final String id, title, thumb;

  WebtoonModel({required this.id, required this.title, required this.thumb});

// Named Constructor
  WebtoonModel.fronJson(Map<String, dynamic> json)
      : title = json['title'],
        thumb = json['thumb'],
        id = json['id'];
}
