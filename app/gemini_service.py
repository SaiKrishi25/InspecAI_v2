import os
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Import database functions for querying detection data
from app.database import DetectionStore, ReportStore, get_db_connection
# Import prompt loader for centralized prompt management
from app.prompt_loader import get_prompt_loader

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize prompt loader
        self.prompts = get_prompt_loader()
        
        print(f"Gemini API initialized successfully")
    
    def _is_data_query(self, message: str) -> bool:
        """Check if the user is asking about detection data"""
        data_keywords = self.prompts.get_data_query_keywords()
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in data_keywords)
    
    def _fetch_detection_data(self, message: str) -> Optional[str]:
        """Fetch relevant detection data based on user query"""
        message_lower = message.lower()
        data_context = ""
        
        try:
            # Get today's date for reference
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Fetch analytics overview
            overview = DetectionStore.get_analytics_overview(30)
            
            # Fetch recent detections (last 10)
            recent_detections = DetectionStore.get_recent_detections(10)
            
            # Fetch defect distribution
            defect_distribution = DetectionStore.get_defect_distribution()
            
            # Build context string using template
            data_context = f"""
=== LIVE DATABASE INFORMATION (as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) ===

ANALYTICS OVERVIEW:
- Total Detections (All Time): {overview['total_detections']}
- Detections Today: {overview['today_detections']}
- Detections This Week: {overview['week_detections']}
- Detections This Month: {overview['month_detections']}
- Total Defects Found: {overview['total_defects']}
- Average Confidence Score: {overview['avg_confidence']:.1%}

DEFECTS BY TYPE:
"""
            for defect_type, count in overview.get('defects_by_type', {}).items():
                data_context += f"- {defect_type}: {count} occurrences\n"
            
            # Add recent detections details
            if recent_detections:
                data_context += "\nRECENT DETECTIONS (Latest 10):\n"
                for det in recent_detections:
                    timestamp = det.get('timestamp', 'N/A')
                    detection_id = det.get('detection_id', 'N/A')
                    image_id = det.get('image_id', 'N/A')
                    total_defects = det.get('total_defects', 0)
                    mode = 'RCNN' if det.get('detection_mode') == 'fasterrcnn_only' else 'SAM2'
                    report_id = det.get('report_id', 'N/A')
                    
                    data_context += f"""
Detection:
  - Timestamp: {timestamp}
  - Detection ID: {detection_id}
  - Image ID: {image_id}
  - Report ID: {report_id}
  - Defects Found: {total_defects}
  - Model Used: {mode}
"""
                    # Get defect details for this detection
                    detection_details = DetectionStore.get_detection_by_id(detection_id)
                    if detection_details and detection_details.get('defects'):
                        data_context += "  - Defect Details:\n"
                        for defect in detection_details['defects']:
                            data_context += f"    * {defect.get('defect_type', 'Unknown')} at {defect.get('location', 'unknown')} ({defect.get('confidence_score', 0):.1%} confidence)\n"
            
            # Add defect distribution
            if defect_distribution:
                data_context += "\nDEFECT TYPE STATISTICS:\n"
                for dist in defect_distribution:
                    data_context += f"- {dist['defect_type']}: {dist['count']} total (avg confidence: {dist['avg_confidence']:.1%})\n"
            
            # Add instructions from prompts
            data_context += self.prompts.get_data_context_instructions()
            
        except Exception as e:
            print(f"[GeminiService] Error fetching detection data: {e}")
            data_context = "\n[Note: Could not fetch live database information at this time.]\n"
        
        return data_context
    
    def get_system_prompt(self, current_page: str) -> str:
        """Get the full system prompt for a specific page"""
        return self.prompts.get_system_prompt(current_page)
    
    def chat(self, messages: List[Dict[str, str]], current_page: str = "dashboard") -> str:
        system_prompt = self.get_system_prompt(current_page)
        
        # Get the latest user message
        latest_message = messages[-1]['content'] if messages else ""
        
        # Check if user is asking about detection data
        data_context = ""
        if self._is_data_query(latest_message):
            data_context = self._fetch_detection_data(latest_message)
        
        conversation_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in messages
        ])
        
        # Build full prompt with optional data context
        full_prompt = f"{system_prompt}"
        if data_context:
            full_prompt += f"\n{data_context}"
        full_prompt += f"\n\nConversation:\n{conversation_text}\n\nAssistant:"
        
        # Get generation config from prompts
        gen_config = self.prompts.get_assistant_generation_config()
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": gen_config.get('temperature', 0.7),
                "top_k": gen_config.get('top_k', 40),
                "top_p": gen_config.get('top_p', 0.95),
                "max_output_tokens": gen_config.get('max_output_tokens', 1024),
            }
        )
        
        if not response.text:
            raise Exception("No response generated from Gemini API")
        
        return response.text
    
    def get_suggested_questions(self, current_page: str) -> List[str]:
        """Get suggested questions for a specific page"""
        return self.prompts.get_suggested_questions(current_page)

gemini_service = None

def get_gemini_service():
    global gemini_service
    if gemini_service is None:
        gemini_service = GeminiService()
    return gemini_service
