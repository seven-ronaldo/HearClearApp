// screens/VoiceScreen.js
import React, { useEffect, useState } from "react";
import { View, Button, Text, Alert, StyleSheet } from "react-native";
import { loadModel, runInference } from "TFLiteService";
import { saveVoiceHistory } from "../scripts/voiceHistory";

export default function VoiceScreen() {
  const [output, setOutput] = useState(null);
  const [userId] = useState("00000000-0000-0000-0000-000000000000"); // Thay bằng user ID thật

  useEffect(() => {
    loadModel()
      .then(() => console.log("Model loaded"))
      .catch((err) => console.error("Model load error:", err));
  }, []);

  const handleRunModel = async () => {
    try {
      // Test input giả lập MFCC (thay bằng array thật)
      const dummyInput = new Array(128).fill(0.1);
      const result = await runInference(dummyInput);

      const decoded = decodeCTCOutput(result); // Hàm decode bạn tự viết (simple greedy decode)
      setOutput(decoded);

      await saveVoiceHistory(userId, decoded);
      Alert.alert("Thành công", "Đã lưu kết quả vào Supabase");
    } catch (error) {
      console.error(error);
      Alert.alert("Lỗi", error.message || "Không thể chạy mô hình");
    }
  };

  return (
    <View style={styles.container}>
      <Button title="Chạy mô hình nhận diện" onPress={handleRunModel} />
      {output && <Text style={styles.text}>Kết quả: {output}</Text>}
    </View>
  );
}

function decodeCTCOutput(tensorOutput) {
  // Simple greedy decoder demo (chỉnh lại theo mô hình bạn)
  const mapping = " abcdefghijklmnopqrstuvwxyz"; // index → char
  const outputIndices = tensorOutput[0].map(Math.round); // ví dụ output là logits
  let result = "";
  let prev = -1;
  for (const i of outputIndices) {
    if (i !== prev && i !== 0) result += mapping[i] || "";
    prev = i;
  }
  return result;
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 10 },
  text: { fontSize: 18, marginTop: 10 },
});
