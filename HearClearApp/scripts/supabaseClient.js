import { createClient } from "@supabase/supabase-js";

const supabaseUrl = "https://xobauyrnhaagodowosmm.supabase.co";
const supabaseKey =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhvYmF1eXJuaGFhZ29kb3dvc21tIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkxODcwOTgsImV4cCI6MjA2NDc2MzA5OH0.MAj2vG5ZrfUBXDQEmEfipI4Q10fQvpMUpt87A0IxCRY";

export const supabase = createClient(supabaseUrl, supabaseKey);
