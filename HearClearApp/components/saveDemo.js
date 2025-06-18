import { supabase } from "./supabaseClient";

async function saveHistory(userId, transcript) {
  const { data, error } = await supabase
    .from("voice_history")
    .insert([{ user_id: userId, transcript }]);

  if (error) {
    console.error("Lỗi khi lưu lịch sử:", error);
  } else {
    console.log("Lưu lịch sử thành công:", data);
  }
}
