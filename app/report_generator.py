#!/usr/bin/env python3
"""
Report Generation Module for Vehicle Inspection Pipeline
Uses Gemini AI for professional, industry-level defect analysis
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
from io import BytesIO

# Gemini AI Integration
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[ReportGenerator] Gemini not available. Install with: pip install google-generativeai")

# Import prompt loader for centralized prompt management
try:
    from app.prompt_loader import get_prompt_loader
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    print("[ReportGenerator] Prompt loader not available, using inline prompts")

class VehicleInspectionReportGenerator:
    """
    Generates comprehensive vehicle inspection reports with Gemini AI analysis
    """
    
    def __init__(self, use_ai_analysis=True):
        self.use_ai_analysis = use_ai_analysis and GEMINI_AVAILABLE
        self.gemini_model = None
        self.prompts = None
        
        if PROMPTS_AVAILABLE:
            self.prompts = get_prompt_loader()
        
        if self.use_ai_analysis:
            self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI for report generation"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("[ReportGenerator] GEMINI_API_KEY not found, using template-based descriptions")
                self.use_ai_analysis = False
                return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print("[ReportGenerator] Gemini AI initialized successfully")
            
        except Exception as e:
            print(f"[ReportGenerator] Failed to initialize Gemini: {e}")
            print("[ReportGenerator] Falling back to template-based descriptions")
            self.use_ai_analysis = False
    
    def generate_unique_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8].upper()
        return f"RPT_{timestamp}_{unique_id}"
    
    def generate_unique_defect_id(self) -> str:
        """Generate unique defect ID"""
        return f"D{uuid.uuid4().hex[:8].upper()}"
    
    def generate_defect_description(self, defect: Dict) -> str:
        """Generate AI-powered professional defect description using Gemini"""
        if self.use_ai_analysis and self.gemini_model:
            try:
                # Get prompt from YAML or use inline fallback
                if self.prompts:
                    prompt = self.prompts.get_defect_analysis_prompt(
                        defect_type=defect['class'],
                        location=defect['location'],
                        confidence=f"{defect['score']:.1%}",
                        bbox=str(defect.get('bbox', 'N/A'))
                    )
                    gen_config = self.prompts.get_defect_analysis_config()
                else:
                    # Fallback to inline prompt if prompts not available
                    prompt = f"""You are an automotive quality control expert. Analyze the following vehicle defect and provide a professional, industry-standard assessment.

DEFECT INFORMATION:
- Type: {defect['class']}
- Location: {defect['location']}
- Confidence Score: {defect['score']:.1%}
- Bounding Box: {defect.get('bbox', 'N/A')}

Provide a concise assessment in exactly this format (3-4 sentences total):

1. DEFECT DESCRIPTION: What is this defect and its characteristics?\n
2. IMPACT ASSESSMENT: How does this affect the vehicle's appearance/integrity?\n
3. RECOMMENDED ACTION: What repair method is recommended?

Keep the response professional, technical, and under 100 words. Do not use bullet points or numbered lists in the response - write in paragraph form."""
                    gen_config = {'temperature': 0.3, 'max_output_tokens': 150}

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": gen_config.get('temperature', 0.3),
                        "max_output_tokens": gen_config.get('max_output_tokens', 150),
                    }
                )
                
                if response.text:
                    return response.text.strip()
                    
            except Exception as e:
                print(f"[ReportGenerator] Gemini generation failed: {e}")
        
        # Fallback to template-based descriptions
        return self._get_template_description(defect)
    
    def _get_template_description(self, defect: Dict) -> str:
        """Template-based professional defect descriptions"""
        defect_type = defect['class'].lower()
        location = defect['location']
        confidence = defect['score']
        severity = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "minor"
        
        # Try to get template from YAML
        if self.prompts:
            template = self.prompts.get_fallback_template(defect_type)
            if template:
                return template.format(
                    severity=severity,
                    location=location,
                    confidence=f"{confidence:.1%}",
                    defect_type=defect_type
                )
        
        # Inline fallback templates
        templates = {
            'dent': f"A {severity}-severity dent has been identified in the {location} area with {confidence:.1%} detection confidence. "
                   f"This structural deformation may affect the vehicle's aesthetic appeal and could indicate underlying panel damage. "
                   f"Recommended repair: Paintless Dent Repair (PDR) for minor dents, or conventional body work with refinishing for deeper impacts.",
            
            'scratch': f"A surface scratch has been detected in the {location} region with {confidence:.1%} confidence. "
                      f"This linear abrasion affects the clear coat and potentially the base paint layer, compromising corrosion protection. "
                      f"Recommended repair: Wet sanding and polishing for superficial scratches, or spot refinishing for deeper marks penetrating the base coat.",
            
            'paint defect': f"A paint defect has been identified in the {location} area with {confidence:.1%} confidence. "
                           f"This anomaly indicates potential issues with paint application, adhesion, or environmental damage. "
                           f"Recommended repair: Professional paint correction or spot refinishing depending on defect depth and extent.",
            
            'water spots': f"Water spot deposits detected in the {location} region with {confidence:.1%} confidence. "
                          f"These mineral deposits result from water evaporation and can etch into the clear coat if left untreated. "
                          f"Recommended repair: Clay bar treatment followed by machine polishing to restore surface clarity.",
            
            'hazing': f"Surface hazing identified in the {location} area with {confidence:.1%} confidence. "
                     f"This cloudy appearance indicates oxidation or micro-scratching of the clear coat layer. "
                     f"Recommended repair: Multi-stage paint correction using compound and polish to restore optical clarity.",
            
            'oxidation': f"Paint oxidation detected in the {location} region with {confidence:.1%} confidence. "
                        f"This degradation of the paint surface results from UV exposure and environmental factors, causing a dull, chalky appearance. "
                        f"Recommended repair: Heavy-cut compound correction followed by sealant application, or respray for severe cases."
        }
        
        return templates.get(defect_type, 
            f"A {defect_type} defect has been detected in the {location} area with {confidence:.1%} confidence. "
            f"This anomaly requires professional assessment to determine appropriate remediation. "
            f"Recommended action: Detailed inspection by qualified technician to evaluate extent and repair options.")
    
    def generate_overall_summary(self, defects: List[Dict], report_data: Dict) -> str:
        """Generate overall inspection summary using Gemini AI"""
        total_defects = len(defects)
        defect_types = list(set([d['class'] for d in defects]))
        detection_mode = report_data.get('detection_mode', 'unknown')
        
        # Calculate severity distribution
        high_severity = sum(1 for d in defects if d['score'] > 0.8)
        medium_severity = sum(1 for d in defects if 0.6 < d['score'] <= 0.8)
        low_severity = sum(1 for d in defects if d['score'] <= 0.6)
        
        # Calculate average confidence
        avg_confidence = sum(d['score'] for d in defects) / total_defects if total_defects > 0 else 0
        
        if self.use_ai_analysis and self.gemini_model and total_defects > 0:
            try:
                # Build defect details
                defect_details = "\n".join([
                    f"- {d['class']} at {d['location']} ({d['score']:.1%} confidence)"
                    for d in defects
                ])
                
                # Determine detection model name
                if 'fasterrcnn' in detection_mode:
                    detection_model = 'Faster R-CNN (Structural Defects)'
                elif 'sam2' in detection_mode:
                    detection_model = 'SAM2 (Surface Defects)'
                else:
                    detection_model = detection_mode
                
                # Get prompt from YAML or use inline fallback
                if self.prompts:
                    prompt = self.prompts.get_overall_summary_prompt(
                        total_defects=total_defects,
                        defect_types=', '.join(defect_types),
                        detection_model=detection_model,
                        high_severity=high_severity,
                        medium_severity=medium_severity,
                        low_severity=low_severity,
                        avg_confidence=f"{avg_confidence:.1%}",
                        defect_details=defect_details
                    )
                    gen_config = self.prompts.get_overall_summary_config()
                else:
                    # Fallback to inline prompt
                    prompt = f"""You are an automotive quality control manager. Provide a professional executive summary for this vehicle inspection report.

INSPECTION DATA:
- Total Defects Found: {total_defects}
- Defect Types: {', '.join(defect_types)}
- Detection Model Used: {detection_model}
- Severity Distribution: {high_severity} High, {medium_severity} Medium, {low_severity} Low
- Average Detection Confidence: {avg_confidence:.1%}

DEFECTS IDENTIFIED:
{defect_details}

Write a professional 3-4 sentence executive summary that includes:
1. Overall quality assessment (Pass/Conditional Pass/Fail)
2. Key findings summary
3. Priority recommendations
4. Estimated repair urgency (Immediate/Scheduled/Monitor)

Keep it concise, professional, and actionable. Do not use bullet points."""
                    gen_config = {'temperature': 0.3, 'max_output_tokens': 200}

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": gen_config.get('temperature', 0.3),
                        "max_output_tokens": gen_config.get('max_output_tokens', 200),
                    }
                )
                
                if response.text:
                    return response.text.strip()
                    
            except Exception as e:
                print(f"[ReportGenerator] Gemini summary generation failed: {e}")
        
        # Fallback template summary
        return self._get_template_summary(defects, total_defects, defect_types, 
                                          high_severity, medium_severity, low_severity, avg_confidence)
    
    def _get_template_summary(self, defects: List[Dict], total_defects: int, 
                              defect_types: List[str], high_severity: int,
                              medium_severity: int, low_severity: int, avg_confidence: float) -> str:
        """Get template-based summary"""
        
        # Try to get from YAML first
        if self.prompts:
            if total_defects == 0:
                template = self.prompts.get_quality_assessment('pass')
                if template:
                    return template
            elif total_defects <= 2 and high_severity == 0:
                template = self.prompts.get_quality_assessment('conditional_pass_minor')
                if template:
                    return template.format(
                        defect_count=total_defects,
                        defect_types=', '.join(defect_types),
                        avg_confidence=f"{avg_confidence:.1%}"
                    )
            elif high_severity > 0:
                template = self.prompts.get_quality_assessment('requires_attention')
                if template:
                    return template.format(
                        defect_count=total_defects,
                        high_severity=high_severity,
                        defect_types=', '.join(defect_types),
                        avg_confidence=f"{avg_confidence:.1%}"
                    )
            else:
                template = self.prompts.get_quality_assessment('conditional_pass_multi')
                if template:
                    return template.format(
                        defect_count=total_defects,
                        defect_type_count=len(defect_types),
                        defect_types=', '.join(defect_types),
                        high_severity=high_severity,
                        medium_severity=medium_severity,
                        low_severity=low_severity
                    )
        
        # Inline fallback
        if total_defects == 0:
            return ("QUALITY ASSESSMENT: PASS. No defects were detected during the automated inspection. "
                   "The vehicle meets quality control standards for surface and structural integrity. "
                   "Recommended action: Proceed with standard delivery protocols.")
        elif total_defects <= 2 and high_severity == 0:
            return (f"QUALITY ASSESSMENT: CONDITIONAL PASS. {total_defects} minor defect(s) identified "
                   f"({', '.join(defect_types)}). Average detection confidence: {avg_confidence:.1%}. "
                   f"These issues are cosmetic and do not affect vehicle functionality. "
                   f"Recommended action: Schedule minor touch-up work before customer delivery.")
        elif high_severity > 0:
            return (f"QUALITY ASSESSMENT: REQUIRES ATTENTION. {total_defects} defect(s) detected with "
                   f"{high_severity} high-severity issue(s) requiring immediate remediation. "
                   f"Defect types: {', '.join(defect_types)}. Average confidence: {avg_confidence:.1%}. "
                   f"Recommended action: Prioritize repair of high-severity defects before release.")
        else:
            return (f"QUALITY ASSESSMENT: CONDITIONAL PASS. {total_defects} defects identified across "
                   f"{len(defect_types)} categories ({', '.join(defect_types)}). "
                   f"Severity distribution: {high_severity} high, {medium_severity} medium, {low_severity} low. "
                   f"Recommended action: Comprehensive quality remediation before customer handover.")
    
    def create_pdf_report(self, report_data: Dict, defects: List[Dict], output_path: str = None) -> str:
        """Create comprehensive PDF report"""
        
        if output_path is None:
            output_path = f"reports/inspection_report_{report_data['id']}.pdf"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "reports", exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkred,
            spaceBefore=15,
            spaceAfter=10
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph("Vehicle Inspection Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Report Information
        elements.append(Paragraph("Report Information", heading_style))
        report_info = [
            ["Report ID:", report_data['id']],
            ["Generated:", report_data['timestamp']],
            ["Image ID:", report_data.get('image_id', 'N/A')],
            ["Total Defects Found:", str(len(defects))]
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        # Vehicle Information
        elements.append(Paragraph("Vehicle Information", heading_style))
        vehicle_info = [
            ["Car Model:", "Standard Vehicle"],
            ["VIN:", f"VIN{uuid.uuid4().hex[:12].upper()}"],
            ["Inspection Date:", report_data['timestamp']],
            ["Inspector:", "AI Inspection System"]
        ]
        
        vehicle_table = Table(vehicle_info, colWidths=[2*inch, 4*inch])
        vehicle_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(vehicle_table)
        elements.append(Spacer(1, 20))
        
        # Defect Summary Table
        if defects:
            elements.append(Paragraph("Defect Summary Table", heading_style))
            
            table_data = [["Defect ID", "Type", "Location", "Confidence", "Severity", "Action Required"]]
            
            for defect in defects:
                severity = "High" if defect['score'] > 0.8 else "Medium" if defect['score'] > 0.6 else "Low"
                action = "Immediate Repair" if severity == "High" else "Monitor" if severity == "Low" else "Schedule Repair"
                
                table_data.append([
                    defect['id'],
                    defect['class'].title(),
                    defect['location'].title(),
                    f"{defect['score']:.1%}",
                    severity,
                    action
                ])
            
            defect_table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1.3*inch, 1*inch, 1*inch, 1.5*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(defect_table)
            elements.append(Spacer(1, 20))
        
        # AI Analysis Section
        elements.append(Paragraph("AI Analysis", heading_style))
        
        if defects:
            for defect in defects:
                elements.append(Paragraph(f"Defect ID: {defect['id']}", styles['Heading3']))
                description = self.generate_defect_description(defect)
                elements.append(Paragraph(description, styles['Normal']))
                elements.append(Spacer(1, 10))
        
        # Overall Summary
        elements.append(Paragraph("Overall Assessment", heading_style))
        summary = self.generate_overall_summary(defects, report_data)
        elements.append(Paragraph(summary, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Inspection Images Section
        if 'original_image_path' in report_data and 'processed_image_path' in report_data:
            elements.append(Paragraph("Inspection Images", heading_style))
            
            try:
                # Original image
                elements.append(Paragraph("Original Image", styles['Heading3']))
                if os.path.exists(report_data['original_image_path']):
                    img = ReportLabImage(report_data['original_image_path'], width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                
                # Processed image with detections
                elements.append(Paragraph("Detection Results", styles['Heading3']))
                if os.path.exists(report_data['processed_image_path']):
                    img = ReportLabImage(report_data['processed_image_path'], width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 10))
                    
            except Exception as e:
                elements.append(Paragraph(f"Error loading images: {str(e)}", styles['Normal']))
        
        # Build PDF
        try:
            doc.build(elements)
            print(f"[ReportGenerator] PDF report generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"[ReportGenerator] Error generating PDF: {e}")
            return None
    
    def create_json_report(self, report_data: Dict, defects: List[Dict]) -> Dict:
        """Create JSON format report for API responses"""
        
        json_report = {
            "report_id": report_data['id'],
            "timestamp": report_data['timestamp'],
            "image_id": report_data.get('image_id', None),
            "total_defects": len(defects),
            "defects": defects,
            "summary": {
                "overall_assessment": self.generate_overall_summary(defects, report_data),
                "defect_types": list(set([d['class'] for d in defects])) if defects else [],
                "severity_distribution": self._calculate_severity_distribution(defects)
            },
            "ai_analysis": {
                "model_used": "Gemini 2.0 Flash" if self.use_ai_analysis else "Template Based",
                "analysis_type": "AI-Powered Professional Assessment" if self.use_ai_analysis else "Standard Template",
                "confidence_threshold": 0.5
            }
        }
        
        return json_report
    
    def _calculate_severity_distribution(self, defects: List[Dict]) -> Dict:
        """Calculate distribution of defect severities"""
        if not defects:
            return {"high": 0, "medium": 0, "low": 0}
        
        high = sum(1 for d in defects if d['score'] > 0.8)
        medium = sum(1 for d in defects if 0.6 < d['score'] <= 0.8)
        low = sum(1 for d in defects if d['score'] <= 0.6)
        
        return {"high": high, "medium": medium, "low": low}


# Example usage and testing
if __name__ == "__main__":
    # Test the report generator
    generator = VehicleInspectionReportGenerator()
    
    # Sample data
    sample_defects = [
        {
            "id": generator.generate_unique_defect_id(),
            "class": "Scratch",
            "score": 0.85,
            "location": "front panel",
            "bbox": [100, 150, 200, 250]
        },
        {
            "id": generator.generate_unique_defect_id(),
            "class": "Dent",
            "score": 0.72,
            "location": "rear door",
            "bbox": [300, 200, 400, 300]
        }
    ]
    
    sample_report_data = {
        "id": generator.generate_unique_report_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_id": "IMG_001"
    }
    
    # Generate reports
    json_report = generator.create_json_report(sample_report_data, sample_defects)
    print("JSON Report:", json.dumps(json_report, indent=2))
    
    pdf_path = generator.create_pdf_report(sample_report_data, sample_defects)
    print(f"PDF Report generated: {pdf_path}")
