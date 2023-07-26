import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { theme } from './color';

export default function App() {
  const [tabWork, setTabWork] = useState(true);
  
  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => setTabWork(true)}>
          <Text style={{...styles.btnText, color: tabWork ? theme.active : theme.passive}}>Travel</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setTabWork(false)}>
          <Text style={{...styles.btnText, color: tabWork ? theme.passive : theme.active}}>Work</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
    paddingHorizontal: 30,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 50,
  },
  btnText: {
    fontSize: 38,
    fontWeight: "600",
    color: theme.gray,
  },
});
