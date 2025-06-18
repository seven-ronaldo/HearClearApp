import React, { useState } from "react";
import { View, Text, Button } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { runInference } from "./model";
import { saveHistory } from "./supabaseClient";

export default function App() {
  const [transcript, setTranscript] = useState("");

  const handlePickFile = async () => {
    const res = await DocumentPicker.getDocumentAsync({ type: "audio/wav" });
    if (res.type === "success") {
      const inputData = await preprocessAudio(res.uri); // Hàm xử lý âm thanh
      const output = await runInference(inputData);
      const text = decodeOutput(output); // Hàm giải mã kết quả
      setTranscript(text);
      await saveHistory("user-id", text);
    }
  };

  return (
    <View>
      <Button title="Chọn tệp âm thanh" onPress={handlePickFile} />
      <Text>Kết quả: {transcript}</Text>
    </View>
  );
}
