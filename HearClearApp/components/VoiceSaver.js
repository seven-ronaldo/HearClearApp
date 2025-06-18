import React, { useState } from "react";
import { View, TextInput, Button, Alert, StyleSheet } from "react-native";
import { saveVoiceHistory } from "../scripts/voiceHistory";

export default function VoiceSaver() {
  const [transcript, setTranscript] = useState("");
  const [userId, setUserId] = useState("00000000-0000-0000-0000-000000000000"); // Thay bằng real user_id

  const handleSave = async () => {
    if (!transcript.trim()) {
      Alert.alert("Thông báo", "Vui lòng nhập nội dung giọng nói.");
      return;
    }

    const result = await saveVoiceHistory(userId, transcript);
    if (result?.success) {
      Alert.alert("Thành công", "Đã lưu lịch sử giọng nói.");
      setTranscript("");
    } else {
      Alert.alert("Thất bại", `Lỗi: ${result?.error || "Không rõ lỗi"}`);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Nhập kết quả giọng nói ở đây..."
        value={transcript}
        onChangeText={setTranscript}
        multiline
      />
      <Button title="Gửi kết quả và lưu vào lịch sử" onPress={handleSave} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 12,
    borderRadius: 8,
    minHeight: 100,
    textAlignVertical: "top",
  },
});
