import React, { useState } from "react";
import {
  View,
  Text,
  Button,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";

export default function MainScreen() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultText, setResultText] = useState("");

  const handlePickFile = async () => {
    const res = await DocumentPicker.getDocumentAsync({ type: "audio/wav" });
    if (res.type === "success") {
      setFile(res);
      runInference(res);
    }
  };

  const runInference = async (res) => {
    setIsLoading(true);
    try {
      // TODO: gọi inference từ .tflite ở đây
      const decodedText = "Xin chào, tôi là HearClearApp!"; // Demo output
      setResultText(decodedText);
    } catch (e) {
      setResultText("Đã xảy ra lỗi khi xử lý âm thanh.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🎧 HearClearApp</Text>
      <TouchableOpacity style={styles.button} onPress={handlePickFile}>
        <Text style={styles.buttonText}>Chọn file .wav</Text>
      </TouchableOpacity>

      {isLoading && <ActivityIndicator size="large" color="#007AFF" />}

      {resultText !== "" && (
        <View style={styles.resultBox}>
          <Text style={styles.resultText}>{resultText}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  title: { fontSize: 26, fontWeight: "bold", marginBottom: 20 },
  button: {
    backgroundColor: "#007AFF",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: { color: "white", fontSize: 16 },
  resultBox: {
    marginTop: 20,
    padding: 16,
    backgroundColor: "#f1f1f1",
    borderRadius: 10,
    width: "100%",
  },
  resultText: { fontSize: 16, color: "#333" },
});
