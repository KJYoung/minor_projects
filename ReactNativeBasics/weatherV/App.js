import { StatusBar } from 'expo-status-bar';
import { Button, StyleSheet, Text, View } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <Text>Hello! Open up App.js to start working on your app!</Text>
      <StatusBar style="auto" />
      <Button title="안녕! 하십니까.." onClick={() => console.log("Console 이다!")}>안녕!</Button>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
