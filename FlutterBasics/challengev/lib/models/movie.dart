class MovieModel {
  final bool adult;
  final String title,
      releaseDate,
      posterPath,
      backdropPath,
      originalLanguage,
      originalTitle,
      overview;
  final List<dynamic> genre;
  final int id;
  final double popularity;

// Named Constructor
  MovieModel.fronJson(Map<String, dynamic> json)
      : title = json['title'],
        releaseDate = json['release_date'],
        posterPath = json['poster_path'],
        backdropPath = json['backdrop_path'],
        originalLanguage = json['original_language'],
        originalTitle = json['original_title'],
        overview = json['overview'],
        genre = json['genre_ids'],
        popularity = json['popularity'],
        adult = json['adult'],
        id = json['id'];
}

// {
// "adult":false,
// "backdrop_path":"/i2GVEvltEu3BXn5crBSxgKuTaca.jpg",
// "genre_ids":[27,9648,53],
// "id":614479,
// "original_language":"en",
// "original_title":"Insidious: The Red Door",
// "overview":"To put their demons to rest once and for all, Josh Lambert and a college-aged Dalton Lambert must go deeper into The Further than ever before, facing their family's dark past and a host of new and more horrifying terrors that lurk behind the red door.",
// "popularity":3512.648,
// "poster_path":"/uS1AIL7I1Ycgs8PTfqUeN6jYNsQ.jpg",
// "release_date":"2023-07-05",
// "title":"Insidious: The Red Door",
// "video":false,
// "vote_average":6.7,"vote_count":520
// }
class MovieDetailModel {
  final String title, about, genre, age;

// Named Constructor
  MovieDetailModel.fronJson(Map<String, dynamic> json)
      : title = json['title'],
        about = json['about'],
        genre = json['genre'],
        age = json['age'];
}

class MovieEpisodeModel {
  final String id, thumb, title, rating, date;

// Named Constructor
  MovieEpisodeModel.fronJson(Map<String, dynamic> json)
      : id = json['id'],
        title = json['title'],
        thumb = json['thumb'],
        rating = json['rating'],
        date = json['date'];
}
