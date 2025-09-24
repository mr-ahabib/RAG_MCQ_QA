import React, { useState, useRef, useEffect } from "react";
import { uploadPDF, askQuestion } from "./api/api";
import "./App.css";
function App() {
  const [fileId, setFileId] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState("qa");
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory, loading]);

  const parseMCQ = (text) => {
    const questions = [];
    const blocks = text.split(/\n\nQ\d+:/);
    blocks.forEach((block, idx) => {
      if (!block.trim()) return;
      const qMatch = block.match(/^(.*?)(\nA\))/s);
      const qText = qMatch ? qMatch[1].trim() : block.trim();

      const options = [];
      const optionRegex = /([A-D])\)\s(.*?)(?=\n[A-D]\)|\nCorrect|\n?$)/gs;
      let optMatch;
      while ((optMatch = optionRegex.exec(block)) !== null) {
        options.push({ key: optMatch[1], text: optMatch[2].trim() });
      }

      const correctMatch = block.match(/Correct:\s*([A-D])/);
      const correct = correctMatch ? correctMatch[1] : null;

      questions.push({
        number: idx,
        question: qText.replace(/^Q\d+:/, "").trim(),
        options,
        correct,
      });
    });
    return questions.length > 0 ? questions : null;
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setChatHistory([]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", mode);

      const res = await uploadPDF(file, mode);
      setFileId(res.file_id);

      const newHistory = [
        {
          type: "system",
          content: res.message,
        },
      ];

      if (res.generated_content) {
        if (mode === "mcq") {
          const mcqs = parseMCQ(res.generated_content);
          newHistory.push({ type: "ai", content: mcqs || res.generated_content });
        } else {
          newHistory.push({ type: "ai", content: res.generated_content });
        }
      }

      setChatHistory(newHistory);
    } catch {
      setChatHistory([{ type: "system", content: "Error uploading PDF" }]);
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!fileId) {
      setChatHistory([{ type: "system", content: "Please upload a PDF first." }]);
      return;
    }
    if (!question.trim()) return;

    const userMessage = { type: "user", content: question };
    setChatHistory((prev) => [...prev, userMessage]);

    setLoading(true);
    const currentQuestion = question;
    setQuestion("");

    try {
      const res = await askQuestion(fileId, currentQuestion);

      let aiMessage;
      if (mode === "mcq") {
        const mcqs = parseMCQ(res.answer || "");
        aiMessage = {
          type: "ai",
          content: mcqs || res.answer || "No answer found.",
        };
      } else {
        aiMessage = {
          type: "ai",
          content: res.answer || "No answer found.",
        };
      }

      setChatHistory((prev) => [...prev, aiMessage]);
    } catch {
      const errorMessage = {
        type: "ai",
        content: "Error fetching answer. Please try again.",
      };
      setChatHistory((prev) => [...prev, errorMessage]);
    }

    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const renderMessageContent = (content) => {
    if (Array.isArray(content)) {
      return (
        <div className="mcq-container">
          {content.map((mcq, idx) => (
            <div key={idx} className="mcq-card">
              <div className="mcq-question">
                Q{idx + 1}: {mcq.question}
              </div>
              <ul className="mcq-options">
                {mcq.options.map((opt) => (
                  <li
                    key={opt.key}
                    className={`mcq-option ${mcq.correct === opt.key ? "correct" : ""}`}
                  >
                    {opt.key}) {opt.text}
                  </li>
                ))}
              </ul>
              {mcq.correct && (
                <div className="mcq-correct">
                  ✅ Correct Answer: {mcq.correct}
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    return <div className="message-text">{content}</div>;
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>PDF RAG QA/MCQ</h1>
      </header>

      <div className="mode-selector">
        <label>Select Mode:</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          disabled={uploading || fileId}
        >
          <option value="qa">Question Answering</option>
          <option value="mcq">Multiple Choice Questions</option>
        </select>
      </div>

      <div ref={chatContainerRef} className="chat-container">
        {chatHistory.length === 0 && (
          <div className="empty-state">
            <p>Upload a PDF to get started</p>
          </div>
        )}
        
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <div className="message-sender">{msg.type === "user" ? "You" : msg.type === "system" ? "System" : "Assistant"}</div>
            <div className="message-content">{renderMessageContent(msg.content)}</div>
          </div>
        ))}
        
        {loading && (
          <div className="message ai">
            <div className="message-sender">Assistant</div>
            <div className="message-content thinking">Thinking...</div>
          </div>
        )}
      </div>

      <div className="input-container">
        <div className="input-row">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question..."
            disabled={loading || !fileId}
          />
          <button 
            onClick={handleAsk} 
            disabled={loading || !question.trim() || !fileId}
            className="ask-button"
          >
            Ask
          </button>
        </div>
        
        <div className="upload-row">
          <input
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
            ref={fileInputRef}
            className="file-input"
          />
          <button 
            onClick={triggerFileUpload} 
            disabled={uploading}
            className="upload-button"
          >
            {uploading ? "Uploading..." : "Upload PDF"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;