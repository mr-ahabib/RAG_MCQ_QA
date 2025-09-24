import axios from "axios";

const BASE_URL = "http://192.168.1.183:8000";

// Upload PDF with mode
export const uploadPDF = async (file, mode) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await axios.post(`${BASE_URL}/upload-pdf/?mode=${mode}`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

// Ask Question
export const askQuestion = async (fileId, question) => {
  const res = await axios.post(`${BASE_URL}/ask-question/`, {
    file_id: fileId,
    question: question,
  });
  return res.data;
};
