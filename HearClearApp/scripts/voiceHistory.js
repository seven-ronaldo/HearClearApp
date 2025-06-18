import { supabase } from "./supabaseClient";

export async function saveVoiceHistory(userId, transcript) {
  const { data, error } = await supabase
    .from("voice_history")
    .insert([{ user_id: userId, transcript }]);

  if (error) {
    console.error("Lỗi khi lưu lịch sử:", error.message);
    return { success: false, error: error.message };
  }

  return { success: true, data };
}
