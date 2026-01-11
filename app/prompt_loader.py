"""
Prompt Loader Module for InspecAI
Centralized management of AI prompts from YAML configuration
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path


class PromptLoader:
    """
    Utility class to load and manage prompts from prompts.yaml
    Provides easy access to all AI prompts used in the application
    """
    
    _instance = None
    _prompts = None
    
    def __new__(cls):
        """Singleton pattern to ensure prompts are loaded only once"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_prompts()
        return cls._instance
    
    def _load_prompts(self):
        """Load prompts from YAML file"""
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            prompts_file = current_dir / "prompts.yaml"
            
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
            
            print("[PromptLoader] Prompts loaded successfully from prompts.yaml")
            
        except FileNotFoundError:
            print("[PromptLoader] ERROR: prompts.yaml not found!")
            self._prompts = {}
        except yaml.YAMLError as e:
            print(f"[PromptLoader] ERROR: Failed to parse prompts.yaml: {e}")
            self._prompts = {}
    
    def reload(self):
        """Reload prompts from file (useful for development)"""
        self._load_prompts()
    
    # ═══════════════════════════════════════════════════════════════
    # ASSISTANT PROMPTS (gemini_service.py)
    # ═══════════════════════════════════════════════════════════════
    
    def get_knowledge_base(self) -> str:
        """Get the main knowledge base prompt"""
        return self._prompts.get('assistant', {}).get('knowledge_base', '')
    
    def get_page_context(self, page: str) -> str:
        """Get context for a specific page"""
        contexts = self._prompts.get('assistant', {}).get('page_contexts', {})
        return contexts.get(page, '')
    
    def get_system_prompt(self, current_page: str) -> str:
        """Get full system prompt for a page (knowledge_base + page_context)"""
        knowledge_base = self.get_knowledge_base()
        page_context = self.get_page_context(current_page)
        
        if page_context:
            return f"{knowledge_base}\n{page_context}"
        return knowledge_base
    
    def get_suggested_questions(self, page: str) -> List[str]:
        """Get suggested questions for a specific page"""
        suggestions = self._prompts.get('assistant', {}).get('suggested_questions', {})
        return suggestions.get(page, suggestions.get('default', []))
    
    def get_data_query_keywords(self) -> List[str]:
        """Get keywords that indicate user is asking about data"""
        return self._prompts.get('assistant', {}).get('data_query_keywords', [])
    
    def get_data_context_template(self) -> str:
        """Get template for database context"""
        return self._prompts.get('assistant', {}).get('data_context_template', '')
    
    def get_data_context_instructions(self) -> str:
        """Get instructions to append after database context"""
        return self._prompts.get('assistant', {}).get('data_context_instructions', '')
    
    def get_assistant_generation_config(self) -> Dict[str, Any]:
        """Get generation config for assistant chat"""
        return self._prompts.get('assistant', {}).get('generation_config', {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.95,
            'max_output_tokens': 1024
        })
    
    # ═══════════════════════════════════════════════════════════════
    # REPORT GENERATOR PROMPTS (report_generator.py)
    # ═══════════════════════════════════════════════════════════════
    
    def get_defect_analysis_prompt(self, defect_type: str, location: str, 
                                    confidence: str, bbox: str) -> str:
        """Get formatted defect analysis prompt"""
        template = self._prompts.get('report_generator', {}).get('defect_analysis', {}).get('prompt_template', '')
        return template.format(
            defect_type=defect_type,
            location=location,
            confidence=confidence,
            bbox=bbox
        )
    
    def get_defect_analysis_config(self) -> Dict[str, Any]:
        """Get generation config for defect analysis"""
        return self._prompts.get('report_generator', {}).get('defect_analysis', {}).get('generation_config', {
            'temperature': 0.3,
            'max_output_tokens': 150
        })
    
    def get_overall_summary_prompt(self, total_defects: int, defect_types: str,
                                    detection_model: str, high_severity: int,
                                    medium_severity: int, low_severity: int,
                                    avg_confidence: str, defect_details: str) -> str:
        """Get formatted overall summary prompt"""
        template = self._prompts.get('report_generator', {}).get('overall_summary', {}).get('prompt_template', '')
        return template.format(
            total_defects=total_defects,
            defect_types=defect_types,
            detection_model=detection_model,
            high_severity=high_severity,
            medium_severity=medium_severity,
            low_severity=low_severity,
            avg_confidence=avg_confidence,
            defect_details=defect_details
        )
    
    def get_overall_summary_config(self) -> Dict[str, Any]:
        """Get generation config for overall summary"""
        return self._prompts.get('report_generator', {}).get('overall_summary', {}).get('generation_config', {
            'temperature': 0.3,
            'max_output_tokens': 200
        })
    
    def get_fallback_template(self, defect_type: str) -> str:
        """Get fallback template for a specific defect type"""
        templates = self._prompts.get('report_generator', {}).get('fallback_templates', {})
        # Normalize the defect type key (e.g., "paint defect" -> "paint_defect")
        normalized_key = defect_type.lower().replace(' ', '_')
        return templates.get(normalized_key, templates.get('default', ''))
    
    def get_quality_assessment(self, assessment_type: str) -> str:
        """Get quality assessment template"""
        assessments = self._prompts.get('report_generator', {}).get('quality_assessments', {})
        return assessments.get(assessment_type, '')
    
    # ═══════════════════════════════════════════════════════════════
    # METADATA
    # ═══════════════════════════════════════════════════════════════
    
    def get_metadata(self) -> Dict[str, str]:
        """Get prompt file metadata"""
        return self._prompts.get('metadata', {})
    
    def get_version(self) -> str:
        """Get prompts version"""
        return self._prompts.get('metadata', {}).get('version', 'unknown')


# Singleton instance for easy import
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the singleton PromptLoader instance"""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader
