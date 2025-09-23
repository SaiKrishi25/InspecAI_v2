import os
import cv2
import numpy as np
from typing import List, Dict

def _color_for_defect(defect_class: str):
    """Get color for defect type"""
    defect_colors = {
        "dent": (0, 0, 255),       # Red
        "scratch": (0, 165, 255),  # Orange
        "crack": (0, 255, 255),    # Yellow
        "rust": (42, 42, 165),     # Brown
        "paint_damage": (255, 0, 255),  # Magenta
        "misalignment": (255, 0, 0),    # Blue for misalignments
    }
    
    if defect_class in defect_colors:
        return defect_colors[defect_class]
    else:
        # Generate random color for unknown defects
        np.random.seed(abs(hash(defect_class)) % (2**32 - 1))
        return (
            np.random.randint(50, 255),
            np.random.randint(50, 255), 
            np.random.randint(50, 255)
        )

def draw_defect_detections(image_path: str, detections: List[Dict], save_path: str):
    """Draw defect detection results on image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        defect_class = det.get("class", "unknown")
        score = det.get("score", 0.0)
        label = f"{defect_class} {score:.2f}"
        color = _color_for_defect(defect_class)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - baseline - 5), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def draw_surface_defect_results(img, surface_defects: List[Dict]):
    """Draw surface defect detection results on image with bounding boxes"""
    
    # Define colors for different surface defect types
    defect_colors = {
        'paint_defect': (0, 0, 255),      # Red
        'contamination': (0, 255, 255),   # Yellow
        'corrosion': (42, 42, 165),       # Brown
        'water_spots': (255, 0, 255)      # Magenta
    }
    
    for i, defect in enumerate(surface_defects):
        defect_type = defect.get('defect_type', 'unknown')
        class_name = defect.get('class', 'Unknown')
        score = defect.get('score', 0.0)
        bbox = defect.get('bbox', [])
        location = defect.get('location', 'unknown_area')
        
        if not bbox or len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        color = defect_colors.get(defect_type, (128, 128, 128))  # Default gray
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Create label with defect type and confidence
        label = f"{class_name} {score:.2f}"
        
        # Draw label background
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - baseline - 5), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add location info at bottom of bounding box
        location_label = f"@{location.replace('_', ' ')}"
        cv2.putText(img, location_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    return img

def draw_inspection_results(image_path: str, results: Dict, save_path: str):
    """Draw complete inspection results (defects + misalignments) on image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Draw defect detections
    defects = results.get("defects", [])
    for det in defects:
        x1, y1, x2, y2 = map(int, det["bbox"])
        defect_class = det.get("class", "unknown")
        score = det.get("score", 0.0)
        label = f"{defect_class} {score:.2f}"
        color = _color_for_defect(defect_class)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - baseline - 5), (x1 + w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw surface defect results
    surface_defects = results.get("surface_defects", [])
    img = draw_surface_defect_results(img, surface_defects)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
