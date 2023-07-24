import { StatusBar } from 'expo-status-bar';
import { useState, useEffect } from 'react';
import { ActivityIndicator, StyleSheet, Text, View, ScrollView, Dimensions } from 'react-native';
import * as Location from 'expo-location';
import { Fontisto } from "@expo/vector-icons";

import API_KEY from './API.js';

const { width: WINDOW_WIDTH, height } = Dimensions.get("window");

/*
  [{"city": "샌프란시스코", "country": "미 합중국", "district": "Union Square", "isoCountryCode": "US", "name": "1 Stockton St", "postalCode": "94108", "region": "CA", "street": "Stockton St", "streetNumber": "1", "subregion": "샌프란시스코", "timezone": "America/Los_Angeles"}]
  [{"city": null, "country": "대한민국", "district": "유성구", "isoCountryCode": "KR", "name": "201", "postalCode": "55413", "region": "대전광역시", "street": "신봉동", "streetNumber": "201", "subregion": null, "timezone": null}]
*/

const icons = {
  Clouds: "cloudy",
  Clear: "day-sunny",
  Atmosphere: "cloudy-gusts",
  Snow: "snow",
  Rain: "rains",
  Drizzle: "rain",
  Thunderstorm: "lightning",
};

export default function App() {
  const [city, setCity] = useState("Loading...");
  const [subcity, setSubCity] = useState("");
  const [location, setLocation] = useState();
  const [dayForecasts, setDayForecasts] = useState([]);
  const [ok, setOk] = useState(true);

  
  const getWeather = async () => {
    const { granted } = await Location.requestForegroundPermissionsAsync();
    if(granted){
      const { coords: { latitude, longitude }} = await Location.getCurrentPositionAsync({ accuracy: 5 });
      const location = await Location.reverseGeocodeAsync({ latitude, longitude }, {useGoogleMaps: false});

      if(location[0].city){
        setCity(location[0].city);
      }else{ // Korea Region System.
        setCity(`${location[0].region}`);
        setSubCity(`${location[0].district} ${location[0].street}`);
      };
      setLocation(location[0]);
      const response = await fetch(`https://api.openweathermap.org/data/2.5/forecast?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&units=metric`);
      const json = await response.json(); // { "city", "cnt", "cod", "list", "message" } type.
      // json.list :  { "clouds", "dt", "dt_txt", "main", "pop", "sys", "viibility", "weather", "wind" } type.
      // json.list.main : { "feels_like", "grnd_level", "humidity", "pressure", "sea_level", "temp", "temp_kf", "temp_max", "temp_min" } type.
      // json.list.weather : { "description", "icon", "id", "main" }[] type.
      const weatherInfos = json.list.map((e) => { return { ...e.main, ...e.weather[0], dt: e.dt, dt_txt: e.dt_txt }});
      
      // Filter For Daily Forecast.
      const dailyWeatherInfos = weatherInfos.filter((wI) => wI.dt_txt.includes("09:00:00"));
      setDayForecasts(dailyWeatherInfos);
    }else{
      setOk(false); // Denied Permission!
    }
  };
  useEffect(() => {
    getWeather();
  }, []);
  return (
    <View style={styles.container}>
      <View style={styles.city}>
        <Text style={styles.cityName1}>{city}</Text>
        <Text style={styles.cityName2}>{subcity}</Text>
      </View>
      <ScrollView
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.weather}
      >
        {dayForecasts.length !== 0 ? dayForecasts.map((dFC, idx) => {
          return <View style={styles.weatherDay} key={idx}>
            <Text style={styles.dayInd}>{new Date(dFC.dt * 1000).toString().substring(0, 10)}</Text>
            <Fontisto style={{marginTop: 40}} name={icons[dFC.main]} size={120} color="white"/>
            <Text style={styles.dayTemp}>{dFC.temp}</Text>
            <Text style={styles.dayDesc}>{dFC.main}</Text>
            <Text style={styles.dayFullDesc}>{dFC.description}</Text>
          </View>
        })
        :
          <View style={styles.weatherDay}>
            <ActivityIndicator color="white" size="large" />
          </View>}
        <View style={{flex: 1}}></View>
      </ScrollView>
      <StatusBar style='dark' />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1, 
    backgroundColor: "darkcyan",
  },
  city: {
    flex: 1, 
    backgroundColor: "orange",
    justifyContent: "center",
    alignItems: "center",
  },
  cityName1: {
    fontSize: 32,
    fontWeight: 500,
    textAlign: "center",
  },
  cityName2: {
    fontSize: 48,
    fontWeight: 500,
    textAlign: "center",
  },
  weather: {
    backgroundColor: "cornflowerblue",
  },
  weatherDay: {
    width: WINDOW_WIDTH,
    alignItems: "center",
    backgroundColor: "darkcyan",
  },
  dayInd: {
    fontSize: 46,
    marginTop: 15,
  },
  dayTemp: {
    fontSize: 96,
    fontWeight: 500,
    marginTop: 50,
    color: "ghostwhite"
  },
  dayDesc: {
    fontSize: 50,
    marginTop: -10,
  },
  dayFullDesc: {
    fontSize: 30,
    marginTop: 10,
  },
});
