#!/usr/bin/env python3
"""
SAM2 Surface Defect Detection Script for Google Colab
=====================================================

Standalone script for detecting surface defects in automotive/industrial images using SAM2.
This script can detect:
- Paint defects (color variations)
- Surface contamination (spots, stains)
- Corrosion/rust patterns
- Water spots and mineral deposits

Usage in Google Colab:
1. Upload this script to Colab
2. Install dependencies (see INSTALLATION section below)
3. Run: python sam2_surface_defect_detector.py --image_path "path/to/your/image.jpg"

Author: InspecAI Pipeline
"""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# INSTALLATION COMMANDS FOR GOOGLE COLAB
# Run these in separate cells before running this script:
"""
# Install PyTorch (if not already installed)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SAM2
!pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install other dependencies
!pip install opencv-python pillow matplotlib numpy tqdm

# Optional: Install ultralytics if you want YOLO integration later
!pip install ultralytics
"""

class SAM2SurfaceDefectDetector:
    """
    Advanced surface defect detection for automotive/industrial inspection using SAM2
    """
    
    def __init__(self, sam2_model: str = None, device: str = None, max_detections: int = 3):
        """
        Initialize the SAM2 Surface Defect Detector
        
        Args:
            sam2_model: Specific SAM2 model to use (e.g., 'facebook/sam2-hiera-tiny')
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            max_detections: Maximum number of detections to return (default: 3)
        """
        self.sam2_model = sam2_model
        self.device = device
        self.max_detections = max_detections
        
        # Define defect detection parameters
        self.defect_types = {
            'paint_defect': {'color_variance_threshold': 800, 'min_area': 100},
            'contamination': {'brightness_threshold': 30, 'min_area': 50},
            'corrosion': {'rust_hue_range': (10, 25), 'min_area': 75},
            'water_spots': {'circularity_threshold': 0.7, 'min_area': 25}
        }
        
        self.mask_generator = None
        self._load_sam2()
    
    def _load_sam2(self):
        """Load SAM2 model with HuggingFace integration"""
        try:
            # Import SAM2 components
            from sam2.build_sam import build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            import torch
            
            # Determine device
            if self.device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device)
            
            print(f"[SAM2SurfaceDefectDetector] Using device: {device}")
            
            # Try to load SAM2 model using HuggingFace
            try:
                print("[SAM2SurfaceDefectDetector] Loading SAM2 from HuggingFace...")
                
                # Use specified model or try different model sizes (smallest first for faster loading)
                if self.sam2_model:
                    hf_models = [self.sam2_model]
                else:
                    hf_models = [
                        "facebook/sam2-hiera-tiny",      # Fastest, good for Colab
                        "facebook/sam2-hiera-small", 
                        "facebook/sam2-hiera-base-plus",
                        "facebook/sam2-hiera-large"     # Best quality but slower
                    ]
                
                sam2_model = None
                for model_id in hf_models:
                    try:
                        print(f"[SAM2SurfaceDefectDetector] Trying {model_id}...")
                        sam2_model = build_sam2_hf(model_id, device=device)
                        print(f"[SAM2SurfaceDefectDetector] Successfully loaded {model_id}")
                        break
                    except Exception as hf_error:
                        print(f"[SAM2SurfaceDefectDetector] Failed {model_id}: {hf_error}")
                        continue
                
                if sam2_model is not None:
                     self.mask_generator = SAM2AutomaticMaskGenerator(
                         sam2_model,
                         points_per_side=16,           # Reduced to generate fewer masks
                         pred_iou_thresh=0.85,         # Higher threshold for better quality masks
                         stability_score_thresh=0.9,   # Higher stability threshold
                         crop_n_layers=0,              # No crops to reduce mask count
                         crop_n_points_downscale_factor=1,
                         min_mask_region_area=500,     # Much larger minimum area to filter small regions
                         points_per_batch=32,          # Smaller batch for memory
                     )
                     print("[SAM2SurfaceDefectDetector] SAM2 mask generator initialized successfully.")
                else:
                     print("[SAM2SurfaceDefectDetector] All SAM2 HuggingFace models failed")
                     self.mask_generator = None
                    
            except Exception as model_error:
                print(f"[SAM2SurfaceDefectDetector] Failed to load SAM2 model: {model_error}")
                self.mask_generator = None
                
        except ImportError as e:
            print(f"[SAM2SurfaceDefectDetector] SAM2 not installed ({e}). Please install with:")
            print("!pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            self.mask_generator = None
        except Exception as e:
            print(f"[SAM2SurfaceDefectDetector] SAM2 setup failed ({e}). Using dummy mode.")
            self.mask_generator = None
    
    def detect_surface_defects(self, image_path: str) -> List[Dict]:
        """
        Main detection function - detects surface defects using SAM2 segmentation
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of detected defects with their properties
        """
        if self.mask_generator is None:
            print("[SAM2SurfaceDefectDetector] SAM2 not available, using dummy detection")
            return self._dummy_surface_defects(image_path)
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[SAM2SurfaceDefectDetector] Could not load image: {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"[SAM2SurfaceDefectDetector] Processing image: {image.shape}")
        
        # Generate masks using SAM2
        print("[SAM2SurfaceDefectDetector] Generating SAM2 masks...")
        masks = self.mask_generator.generate(image_rgb)
        print(f"[SAM2SurfaceDefectDetector] Generated {len(masks)} masks")
        
        # Analyze each mask for surface defects
        surface_defects = self._analyze_sam2_detections(image, masks)
        print(f"[SAM2SurfaceDefectDetector] Found {len(surface_defects)} potential defects")
        
        return surface_defects
    
    def _analyze_sam2_detections(self, image: np.ndarray, sam2_masks: List[Dict]) -> List[Dict]:
        """Analyze SAM2 detections and classify surface defects"""
        classified_defects = []
        
        # Sort masks by area (largest first) to prioritize significant regions
        sam2_masks_sorted = sorted(sam2_masks, key=lambda x: x.get('area', 0), reverse=True)
        
        for i, mask_data in enumerate(sam2_masks_sorted):
            mask = mask_data.get('segmentation')
            bbox = mask_data.get('bbox')  # [x, y, w, h]
            area = mask_data.get('area', 0)
            
            if mask is None or area < 500:  # Skip small regions (increased threshold)
                continue
            
            # Convert bbox format and extract ROI
            if bbox:
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                roi = self._extract_roi(image, mask, (x, y, x+w, y+h))
                
                if roi is None or roi.size == 0:
                    continue
                
                # Analyze for different defect types
                defect_results = self._classify_defect(roi, mask, bbox)
                
                if defect_results:
                    defect_results.update({
                        'bbox': [int(x), int(y), int(x+w), int(y+h)],
                        'area': area,
                        'location': self._get_location_name(x + w/2, y + h/2, image.shape[1], image.shape[0]),
                        'mask_id': i
                    })
                    classified_defects.append(defect_results)
        
        # Filter and return only the most confident detections
        return self._filter_best_detections(classified_defects)
    
    def _filter_best_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections to keep only the most significant ones"""
        if not detections:
            return detections
        
        # Sort by confidence score (highest first)
        detections_sorted = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Keep only high-confidence detections
        min_confidence = 0.5  # Minimum confidence threshold
        high_confidence_detections = [d for d in detections_sorted if d['score'] >= min_confidence]
        
        # If we have too many detections, keep only the top ones
        if len(high_confidence_detections) > self.max_detections:
            print(f"[SAM2SurfaceDefectDetector] Filtering {len(high_confidence_detections)} detections to top {self.max_detections}")
            return high_confidence_detections[:self.max_detections]
        
        return high_confidence_detections
    
    def _extract_roi(self, image: np.ndarray, mask, bbox) -> np.ndarray:
        """Extract region of interest from image using mask and bbox"""
        try:
            if bbox:
                x1, y1, x2, y2 = bbox
                # Ensure bounds are within image
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                roi = image[y1:y2, x1:x2]
                
                # Apply mask if available and compatible
                if isinstance(mask, np.ndarray) and roi.shape[0] > 0 and roi.shape[1] > 0:
                    mask_roi = mask[y1:y2, x1:x2]
                    if mask_roi.shape[:2] == roi.shape[:2]:
                        roi = cv2.bitwise_and(roi, roi, mask=mask_roi.astype(np.uint8))
                
                return roi
            return None
        except Exception as e:
            print(f"[SAM2SurfaceDefectDetector] Error extracting ROI: {e}")
            return None
    
    def _classify_defect(self, roi: np.ndarray, mask, bbox) -> Dict:
        """Classify the type of surface defect"""
        if roi is None or roi.size == 0:
            return None
        
        defect_scores = {}
        
        # 1. Paint Defects (color variations)
        paint_score = self._detect_paint_defect(roi)
        if paint_score > 0.5:  # Increased threshold
            defect_scores['paint_defect'] = paint_score
        
        # 2. Surface Contamination (includes water spots)
        contamination_score = self._detect_contamination(roi)
        if contamination_score > 0.4:  # Increased threshold
            defect_scores['contamination'] = contamination_score
        
        # 3. Corrosion/Rust
        rust_score = self._detect_corrosion(roi)
        if rust_score > 0.5:  # Increased threshold
            defect_scores['corrosion'] = rust_score
        
        # 4. Water Spots - Enhanced detection
        water_spot_score = self._detect_water_spots(roi, mask)
        if water_spot_score > 0.3:  # Increased threshold
            defect_scores['water_spots'] = water_spot_score
        
        # 5. Simple water spot detection based on brightness patterns
        simple_water_score = self._detect_simple_water_spots(roi)
        if simple_water_score > 0.4:  # Increased threshold
            defect_scores['water_spots'] = max(defect_scores.get('water_spots', 0), simple_water_score)
        
        # Return the highest scoring defect type
        if defect_scores:
            best_defect = max(defect_scores.items(), key=lambda x: x[1])
            return {
                'class': best_defect[0].replace('_', ' ').title(),
                'score': best_defect[1],
                'defect_type': best_defect[0],
                'all_scores': defect_scores
            }
        
        return None
    
    def _detect_paint_defect(self, roi: np.ndarray) -> float:
        """Detect paint defects based on color variations"""
        try:
            # Convert to LAB color space for better color analysis
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Calculate color variance across the region
            color_variance = np.var(lab, axis=(0, 1))
            total_variance = np.sum(color_variance)
            
            # Normalize score (higher variance = more likely paint defect)
            score = min(total_variance / 1000.0, 1.0)
            
            return score
        except:
            return 0.0
    
    def _detect_contamination(self, roi: np.ndarray) -> float:
        """Detect surface contamination (spots, stains, etc.)"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Look for unusual brightness patterns
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Check for spots that are significantly different from surroundings
            bright_spots = np.sum(gray > mean_brightness + 2 * brightness_std)
            dark_spots = np.sum(gray < mean_brightness - 2 * brightness_std)
            
            total_pixels = gray.size
            contamination_ratio = (bright_spots + dark_spots) / total_pixels
            
            return min(contamination_ratio * 5, 1.0)
        except:
            return 0.0
    
    def _detect_corrosion(self, roi: np.ndarray) -> float:
        """Detect rust and corrosion patterns"""
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define rust color ranges in HSV
            rust_ranges = [
                (np.array([10, 50, 50]), np.array([25, 255, 255])),  # Orange-brown
                (np.array([0, 50, 50]), np.array([10, 255, 255]))    # Red-brown
            ]
            
            rust_pixels = 0
            for lower, upper in rust_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                rust_pixels += np.sum(mask > 0)
            
            total_pixels = roi.shape[0] * roi.shape[1]
            rust_ratio = rust_pixels / total_pixels
            
            return min(rust_ratio * 3, 1.0)
        except:
            return 0.0
    
    def _detect_water_spots(self, roi: np.ndarray, mask) -> float:
        """Detect circular water spots and mineral deposits"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Use HoughCircles to detect circular patterns
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                param1=50, param2=30, minRadius=5, maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Check if detected circles have water spot characteristics
                water_spot_count = 0
                for (x, y, r) in circles:
                    # Extract circular region
                    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    
                    circular_roi = cv2.bitwise_and(gray, gray, mask=circle_mask)
                    
                    # Water spots typically have ring-like patterns
                    edge_pixels = cv2.Canny(circular_roi, 50, 150)
                    if np.sum(edge_pixels > 0) > r:  # Significant edge content
                        water_spot_count += 1
                
                return min(water_spot_count / 3.0, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    def _detect_simple_water_spots(self, roi: np.ndarray) -> float:
        """Simple water spot detection based on brightness contrast"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Water spots are typically brighter than surrounding area
            mean_brightness = np.mean(gray)
            
            # Find bright spots (potential water spots)
            bright_threshold = mean_brightness + np.std(gray) * 1.5
            bright_spots = gray > bright_threshold
            
            # Count connected components (individual spots)
            num_labels, labels = cv2.connectedComponents(bright_spots.astype(np.uint8))
            
            if num_labels > 1:  # More than just background
                # Calculate the ratio of bright pixels
                bright_ratio = np.sum(bright_spots) / gray.size
                
                # More spots = higher score
                spot_density_score = min((num_labels - 1) / 20.0, 1.0)
                brightness_score = min(bright_ratio * 10, 1.0)
                
                return (spot_density_score + brightness_score) / 2
            
            return 0.0
        except:
            return 0.0
    
    def _get_location_name(self, x: float, y: float, img_width: int, img_height: int) -> str:
        """Generate location names based on position"""
        if y < img_height * 0.33:
            vertical = "upper"
        elif y < img_height * 0.67:
            vertical = "middle"
        else:
            vertical = "lower"
        
        if x < img_width * 0.33:
            horizontal = "left"
        elif x < img_width * 0.67:
            horizontal = "center"
        else:
            horizontal = "right"
        
        return f"{vertical}_{horizontal}_area"
    
    def _dummy_surface_defects(self, image_path: str) -> List[Dict]:
        """Fallback dummy detection when SAM2 is not available"""
        print("[SAM2SurfaceDefectDetector] Using dummy surface defect detection")
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        
        return [
            {
                "class": "Paint Defect",
                "score": 0.75,
                "defect_type": "paint_defect",
                "bbox": [w//4, h//4, w//2, h//2],
                "location": "middle_center_area",
                "area": (w//4) * (h//4),
                "all_scores": {"paint_defect": 0.75}
            },
            {
                "class": "Water Spots",
                "score": 0.60,
                "defect_type": "water_spots",
                "bbox": [3*w//4, h//6, w-50, h//3],
                "location": "upper_right_area",
                "area": (w//4) * (h//6),
                "all_scores": {"water_spots": 0.60}
            }
        ]

def visualize_results(image_path: str, detections: List[Dict], save_path: str = None):
    """
    Visualize detection results with bounding boxes and labels
    
    Args:
        image_path: Path to the original image
        detections: List of detection results
        save_path: Optional path to save the visualization
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"SAM2 Surface Defect Detection Results\nFound {len(detections)} defects", fontsize=14)
    
    # Color map for different defect types
    color_map = {
        'paint_defect': 'red',
        'contamination': 'orange',
        'corrosion': 'brown',
        'water_spots': 'blue'
    }
    
    # Draw bounding boxes and labels
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        defect_type = detection.get('defect_type', 'unknown')
        color = color_map.get(defect_type, 'green')
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{detection['class']}: {detection['score']:.2f}"
        ax.text(x1, y1-5, label, fontsize=10, color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def print_detection_summary(detections: List[Dict]):
    """Print a summary of detected defects"""
    print("\n" + "="*60)
    print("SURFACE DEFECT DETECTION SUMMARY")
    print("="*60)
    
    if not detections:
        print("No surface defects detected.")
        return
    
    print(f"Total defects found: {len(detections)}")
    print()
    
    # Group by defect type
    defect_counts = {}
    for detection in detections:
        defect_type = detection.get('defect_type', 'unknown')
        defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
    
    print("Defect type breakdown:")
    for defect_type, count in defect_counts.items():
        print(f"  - {defect_type.replace('_', ' ').title()}: {count}")
    
    print("\nDetailed results:")
    print("-" * 40)
    
    for i, detection in enumerate(detections, 1):
        print(f"{i}. {detection['class']}")
        print(f"   Confidence: {detection['score']:.3f}")
        print(f"   Location: {detection['location'].replace('_', ' ').title()}")
        print(f"   Area: {detection.get('area', 'N/A')} pixels")
        
        # Show all scores if available
        if 'all_scores' in detection:
            print("   Defect scores:")
            for defect, score in detection['all_scores'].items():
                print(f"     - {defect.replace('_', ' ').title()}: {score:.3f}")
        print()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='SAM2 Surface Defect Detection')
    parser.add_argument('--image_path', required=True, help='Path to the input image')
    parser.add_argument('--sam2_model', default=None, help='Specific SAM2 model to use')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--save_visualization', default=None, help='Path to save visualization')
    parser.add_argument('--show_plot', action='store_true', help='Show matplotlib plot')
    parser.add_argument('--max_detections', type=int, default=3, help='Maximum number of detections to return (default: 3)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    print("SAM2 Surface Defect Detection")
    print("=" * 40)
    print(f"Input image: {args.image_path}")
    print(f"SAM2 model: {args.sam2_model or 'Auto-detect'}")
    print(f"Device: {args.device or 'Auto-detect'}")
    print()
    
    # Initialize detector
    detector = SAM2SurfaceDefectDetector(
        sam2_model=args.sam2_model,
        device=args.device,
        max_detections=args.max_detections
    )
    
    # Detect defects
    print("Starting surface defect detection...")
    detections = detector.detect_surface_defects(args.image_path)
    
    # Print summary
    print_detection_summary(detections)
    
    # Visualize results
    if detections or args.show_plot:
        save_path = args.save_visualization
        if save_path is None and not args.show_plot:
            # Auto-generate save path
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            save_path = f"{base_name}_surface_defects.png"
        
        visualize_results(args.image_path, detections, save_path)

if __name__ == "__main__":
    main()
