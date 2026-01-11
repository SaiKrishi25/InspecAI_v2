import axios from 'axios';

const BACKEND_API_URL = 'http://localhost:8000';

console.log('AI Assistant initialized - using backend API');

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export const sendMessageToGemini = async (
  messages: Message[],
  currentPage: string
): Promise<string> => {
  try {
    const response = await axios.post(
      `${BACKEND_API_URL}/api/chat`,
      {
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        currentPage: currentPage
      },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (response.data.status === 'error') {
      throw new Error(response.data.error || 'Failed to get response from AI');
    }

    const generatedText = response.data.response;
    
    if (!generatedText) {
      throw new Error('No response generated from Gemini AI');
    }

    return generatedText;
  } catch (error) {
    console.error('Error calling Gemini AI:', error);
    if (axios.isAxiosError(error)) {
      if (error.code === 'ERR_NETWORK') {
        throw new Error('Cannot connect to backend. Make sure backend is running on port 8000.');
      }
      throw new Error(error.response?.data?.error || 'Failed to get response from AI');
    }
    throw error;
  }
};

// Suggested starter questions based on page
export const getSuggestedQuestions = (currentPage: string): string[] => {
  const suggestions: Record<string, string[]> = {
    dashboard: [
      "What do the detection trends tell me?",
      "How is the defect distribution calculated?",
      "What does the confidence score mean?",
      "How can I improve detection accuracy?",
    ],
    infer: [
      "How do I capture an image?",
      "What's the difference between Faster R-CNN and SAM2?",
      "How do I interpret the detection results?",
      "Can I adjust the detection sensitivity?",
    ],
    reports: [
      "How do I search for specific reports?",
      "What information is in the PDF reports?",
      "How are defects categorized?",
      "Can I export report data?",
    ],
  };

  return suggestions[currentPage] || [
    "What is InspecAI?",
    "How does defect detection work?",
    "How do I get started?",
  ];
};
