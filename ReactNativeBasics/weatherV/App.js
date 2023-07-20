import { StatusBar } from 'expo-status-bar';
import { Button, StyleSheet, Text, View } from 'react-native';

export default function App() {
  return (
    <View style={{
      flex: 1,
      backgroundColor: 'white',
      alignItems: 'center',
      justifyContent: 'center',
      border: "1px green dashed",
    }}>
      <Text style={styles.text}>Hllo! Open up App.js to start working on your app!</Text>
      <StatusBar style="dark" />
      <Button title="안녕!! 하십니까.." onClick={() => console.log("Console 이다!")}>안녕!</Button>
    </View>
  );
}

const styles = StyleSheet.create({
  text: {
    fontSize: 44,
    color: "black"
  }
});
