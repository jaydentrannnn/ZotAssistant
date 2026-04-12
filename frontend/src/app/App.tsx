import { useState, useRef, useEffect } from "react";
import { Paperclip, Send, X } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ChatMessage } from "./components/ChatMessage";
import { TypingIndicator } from "./components/TypingIndicator";
import { EmptyState } from "./components/EmptyState";
import { SessionBanner } from "./components/SessionBanner";
import logoImg from "../imports/ZotAssistantLogo.png";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export default function App() {
  // sessionId is generated once per page load — refreshing the page creates a
  // new UUID, which starts a fresh LangChain session server-side.
  const [sessionId] = useState(() => crypto.randomUUID());

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBanner, setShowBanner] = useState(true);
  const [showEmptyInputShake, setShowEmptyInputShake] = useState(false);

  const chatWindowRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const newHeight = Math.min(textareaRef.current.scrollHeight, 120); // max 5 lines ~120px
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [input]);

  const formatTimestamp = () => {
    const now = new Date();
    return now.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  };

  const handleSendMessage = async () => {
    if (!input.trim() && !file) {
      setShowEmptyInputShake(true);
      setTimeout(() => setShowEmptyInputShake(false), 500);
      return;
    }

    // Capture before clearing state
    const messageToSend = input.trim();

    const attachedFile = file;
    const userMessageContent = messageToSend
      ? attachedFile
        ? `${messageToSend}\n\n📎 ${attachedFile.name}`
        : messageToSend
      : `📎 ${attachedFile?.name}`;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: userMessageContent,
      timestamp: formatTimestamp(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setFile(null);
    setError(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("message", messageToSend);
      formData.append("session_id", sessionId);
      if (attachedFile) formData.append("file", attachedFile);

      const response = await fetch("/api/chat", {
        method: "POST",
        // Do NOT set Content-Type — the browser sets multipart/form-data + boundary automatically
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const assistantId = (Date.now() + 1).toString();
      let assistantContent = "";
      let firstChunk = true;

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        for (const line of text.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") break outer;

          // Add the assistant bubble on the very first token so the typing
          // indicator disappears and streaming text appears immediately.
          if (firstChunk) {
            firstChunk = false;
            setIsLoading(false);
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                role: "assistant" as const,
                content: "",
                timestamp: formatTimestamp(),
              },
            ]);
          }

          // Unescape newlines that were escaped for SSE transport
          assistantContent += data.replace(/\\n/g, "\n");
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.id === assistantId) {
              updated[updated.length - 1] = {
                ...last,
                content: assistantContent,
              };
            }
            return updated;
          });
        }
      }

      // Guard: if the stream ended without any chunks, clear the spinner
      if (firstChunk) {
        setIsLoading(false);
      }
    } catch {
      setIsLoading(false);
      setError("Something went wrong connecting to the server. Please try again.");
      setTimeout(() => setError(null), 4000);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // File size validation (10MB limit)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit");
      setTimeout(() => setError(null), 4000);
      return;
    }

    // File type validation
    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
    ];

    if (!allowedTypes.includes(selectedFile.type)) {
      setError(
        "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
      );
      setTimeout(() => setError(null), 4000);
      return;
    }

    setFile(selectedFile);
    setError(null);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    textareaRef.current?.focus();
  };

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden"
      style={{ fontFamily: "Inter, system-ui, sans-serif" }}
    >
      {/* Header */}
      <header className="h-16 bg-white border-b-2 border-[#0064A4] flex items-center justify-between px-4 md:px-6 flex-shrink-0">
        <div className="flex items-center gap-3">
          <img
            src={logoImg}
            alt="ZotAssistant Logo"
            className="h-10 w-auto object-contain"
          />
        </div>

        <h1 className="text-[#003764] font-bold text-base md:text-lg absolute left-1/2 transform -translate-x-1/2 hidden sm:block">
          ZotAssistant
        </h1>

        <div className="px-3 py-1.5 bg-[#F5F7FA] rounded-full text-xs text-[#6B7280] hidden md:block">
          New session on every refresh
        </div>
      </header>

      {/* Session Banner */}
      <AnimatePresence>
        {showBanner && <SessionBanner onDismiss={() => setShowBanner(false)} />}
      </AnimatePresence>

      {/* Chat Window */}
      <div
        ref={chatWindowRef}
        role="log"
        aria-live="polite"
        className="flex-1 overflow-y-auto bg-[#F5F7FA] px-4 md:px-6 py-8 custom-scrollbar"
      >
        <div className="max-w-[760px] mx-auto md:px-6">
          {messages.length === 0 && !isLoading ? (
            <EmptyState onSuggestionClick={handleSuggestionClick} />
          ) : (
            <>
              {messages.map((message) => (
                <ChatMessage key={message.id} {...message} />
              ))}
              {isLoading && <TypingIndicator />}
            </>
          )}
        </div>
      </div>

      {/* Error Toast */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-20 right-4 md:right-6 bg-red-50 border-l-4 border-red-500 px-4 py-3 rounded shadow-lg max-w-[calc(100%-2rem)] md:max-w-md z-50"
          >
            <div className="flex items-start gap-3">
              <p className="text-sm text-red-800">{error}</p>
              <button
                onClick={() => setError(null)}
                className="text-red-500 hover:text-red-700"
                aria-label="Dismiss error"
              >
                <X size={16} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Bar */}
      <div className="h-20 bg-white border-t shadow-[0_-2px_10px_rgba(0,0,0,0.05)] flex items-center px-4 md:px-6 flex-shrink-0">
        <div className="max-w-[760px] mx-auto w-full flex items-end gap-3 md:px-6">
          {/* File Upload */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="mb-2 p-2 text-[#6B7280] hover:text-[#0064A4] hover:bg-[#F5F7FA] rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-[#FFD200]"
            aria-label="Attach file"
          >
            <Paperclip size={20} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.docx,.txt"
            onChange={handleFileSelect}
          />

          {/* Textarea Container */}
          <div className="flex-1 relative">
            {/* File Chip */}
            {file && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute bottom-full mb-2 flex items-center gap-2 bg-[#E5F2FF] border border-[#0064A4] rounded-lg px-3 py-1.5"
              >
                <Paperclip size={14} className="text-[#0064A4]" />
                <span className="text-sm text-[#003764]">{file.name}</span>
                <button
                  onClick={() => setFile(null)}
                  className="text-[#6B7280] hover:text-[#003764]"
                  aria-label="Remove file"
                >
                  <X size={14} />
                </button>
              </motion.div>
            )}

            <motion.textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about courses, policies, or requirements…"
              rows={1}
              animate={
                showEmptyInputShake
                  ? {
                      x: [0, -10, 10, -10, 10, 0],
                    }
                  : {}
              }
              transition={{ duration: 0.4 }}
              className="w-full resize-none border border-[#E5E7EB] rounded-xl px-4 py-3 focus:outline-none focus:border-[#0064A4] focus:ring-2 focus:ring-[#0064A4]/20 transition-colors custom-scrollbar"
              style={{ maxHeight: "120px", minHeight: "48px" }}
            />
          </div>

          {/* Send Button */}
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() && !file}
            className={`mb-2 w-11 h-11 rounded-full flex items-center justify-center transition-colors focus:outline-none focus:ring-2 focus:ring-[#FFD200] ${
              input.trim() || file
                ? "bg-[#0064A4] text-white hover:bg-[#003764]"
                : "bg-[#E5E7EB] text-[#9CA3AF] cursor-not-allowed"
            }`}
            aria-label="Send message"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
