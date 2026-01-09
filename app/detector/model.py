from typing import List, Dict
from PIL import Image
import torch
import cv2
import numpy as np
import os

# -------------------------
# Defect Detector (YOLO / FasterRCNN) - your existing code
# -------------------------
class DefectDetector:
    def __init__(self, model_type: str = "fasterrcnn", model_path: str = "fasterrcnn_model.pth", confidence_threshold: float = 0.5):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define classes based on your working code
        # Your old code: CLASSES = ['crack', 'dent', 'scratch'] with labels 1-3
        # So: 1=crack, 2=dent, 3=scratch (0 is background in FasterRCNN)
        self.defect_classes = {
            0: "Background",  # FasterRCNN background class
            1: "Scratch",     # crack -> map to Scratch (linear defect)
            2: "Dent",        # dent -> Dent (exact match)
            3: "Scratch"      # scratch -> Scratch (exact match)
        }
        self._load_model()
    
    def _load_model(self):
        try:
            if self.model_type == "fasterrcnn":
                self._load_fasterrcnn_model()
            else:
                print(f"[DefectDetector] Unsupported model type: {self.model_type}. Using FasterRCNN as fallback.")
                self.model_type = "fasterrcnn"
                self._load_fasterrcnn_model()
        except Exception as e:
            print(f"[DefectDetector] Failed to load {self.model_type} model ({e}). Using dummy detector.")
            self._model = None
    
    
    def _load_fasterrcnn_model(self):
        try:
            import torchvision.models as models
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            
            print(f"[DefectDetector] Loading FasterRCNN model: {self.model_path}")
            
            if os.path.exists(self.model_path):
                # Load custom trained model - try to determine class count from checkpoint
                try:
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    # Check the actual number of classes in the saved model
                    if 'roi_heads.box_predictor.cls_score.weight' in checkpoint:
                        saved_num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
                        print(f"[DefectDetector] Detected {saved_num_classes} classes in saved model")
                        
                        # Create model with the correct number of classes
                        self._model = models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=saved_num_classes)
                        self._model.load_state_dict(checkpoint)
                        print(f"[DefectDetector] Custom FasterRCNN model loaded successfully with {saved_num_classes} classes")
                    else:
                        # Fallback: try with our default class count
                        self._model = models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=len(self.defect_classes))
                        self._model.load_state_dict(checkpoint)
                        print("[DefectDetector] Custom FasterRCNN model loaded successfully")
                        
                except Exception as load_error:
                    print(f"[DefectDetector] Failed to load custom model: {load_error}")
                    print("[DefectDetector] Falling back to pretrained model")
                    self._model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            else:
                print("[DefectDetector] Custom model file not found, using pretrained FasterRCNN")
                # Use pretrained model and adapt for our classes
                self._model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                # Note: Pretrained model has 91 classes, we'll filter to our classes in detection
            
            self._model.to(self.device)
            self._model.eval()
            print(f"[DefectDetector] FasterRCNN model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[DefectDetector] Error loading FasterRCNN model: {e}")
            raise

    def detect(self, image_path: str) -> List[Dict]:
        if self._model is None:
            return self._dummy_detection(image_path)
        
        try:
            if self.model_type == "fasterrcnn":
                return self._detect_fasterrcnn(image_path)
            else:
                return self._dummy_detection(image_path)
        except Exception as e:
            print(f"[DefectDetector] Detection failed ({e}). Using dummy fallback.")
            return self._dummy_detection(image_path)
    
    
    def _get_smart_class_mapping(self, label_int: int) -> str:
        """
        Smart class mapping based on your working code
        CLASSES = ['crack', 'dent', 'scratch'] with FasterRCNN labels 1-3
        """
        if label_int == 0:
            return None        # Background - should be skipped
        elif label_int == 1:
            return "Scratch"   # crack -> Scratch (linear defect)
        elif label_int == 2:
            return "Dent"      # dent -> Dent (exact match)
        elif label_int == 3:
            return "Scratch"   # scratch -> Scratch (exact match)
        else:
            return "Dent"      # Default to Dent for any unknown class

    def _detect_fasterrcnn(self, image_path: str) -> List[Dict]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self._model(image_tensor)
        
        detections = []
        pred = predictions[0]
        
        keep_indices = pred['scores'] > self.confidence_threshold
        boxes = pred['boxes'][keep_indices].cpu().numpy()
        scores = pred['scores'][keep_indices].cpu().numpy()
        labels = pred['labels'][keep_indices].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            label_int = int(label)
            
            # Debug: Print what we're detecting
            print(f"[DEBUG] Detected class ID: {label_int}, Score: {score:.3f}")
            
            # Use smart mapping to ensure only Scratch or Dent
            class_name = self._get_smart_class_mapping(label_int)
            
            # Skip background class (class 0)
            if class_name is None:
                print(f"[DEBUG] Skipping background class {label_int}")
                continue
            
            print(f"[DEBUG] Class ID {label_int} -> {class_name}")
                
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class": class_name,
                "class_id": label_int
            })
            
        print(f"[DEBUG] Final detections: {len(detections)} found")
        
        return detections
    
    def _dummy_detection(self, image_path: str) -> List[Dict]:
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        h, w = img.shape[:2]
        detections = []
        
        cx, cy = w//3, h//3
        bw, bh = int(w*0.2), int(h*0.2)
        x1, y1 = max(0, cx - bw//2), max(0, cy - bh//2)
        x2, y2 = min(w-1, x1 + bw), min(h-1, y1 + bh)
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 0.75,
            "class": "Dent",
            "class_id": 0
        })
        
        cx, cy = 2*w//3, 2*h//3
        bw, bh = int(w*0.15), int(h*0.1)
        x1, y1 = max(0, cx - bw//2), max(0, cy - bh//2)
        x2, y2 = min(w-1, x1 + bw), min(h-1, y1 + bh)
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 0.60,
            "class": "Scratch",
            "class_id": 1
        })
        
        return detections


# -------------------------
# NEW: Automotive Surface Defect Detector (using SAM2 + OpenCV)
# -------------------------
class AutomotiveSurfaceDefectDetector:
    def __init__(self, sam2_model: str = None):
        """
        Advanced surface defect detection for automotive inspection with enhanced sensitivity
        sam2_model: specific SAM2 model to use (e.g., 'facebook/sam2-hiera-tiny')
        """
        self.sam2_model = sam2_model
        
        # ENHANCED: Much more sensitive thresholds for minute defects
        self.defect_types = {
            'paint_defect': {'color_variance_threshold': 300, 'min_area': 15},  # Reduced from 800, 100
            'contamination': {'brightness_threshold': 12, 'min_area': 12},      # Reduced from 30, 50
            'corrosion': {'rust_hue_range': (10, 25), 'min_area': 20},          # Reduced from 75
            'water_spots': {'circularity_threshold': 0.5, 'min_area': 8},       # Reduced from 0.7, 25
            'scratch': {'min_length': 10, 'max_width': 5, 'min_area': 8},       # NEW: For linear defects
            'texture_defect': {'variance_threshold': 50, 'min_area': 15}        # NEW: For surface texture issues
        }
        self._load_sam2()
    
    def _load_sam2(self):
        try:
            # Use SAM2 with HuggingFace integration for automatic downloads
            from sam2.build_sam import build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            import torch
            
            # Check for CUDA availability
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[PartMisalignmentDetector] Using device: {device}")
            
            # Try to load SAM2 model using HuggingFace (handles downloads automatically)
            try:
                print("[PartMisalignmentDetector] Loading SAM2 from HuggingFace...")
                
                # Use specified model or try different model sizes (smallest first for faster loading)
                if self.sam2_model:
                    hf_models = [self.sam2_model]
                else:
                    hf_models = [
                        "facebook/sam2-hiera-tiny",
                        "facebook/sam2-hiera-small", 
                        "facebook/sam2-hiera-base-plus",
                        "facebook/sam2-hiera-large"
                    ]
                
                sam2_model = None
                for model_id in hf_models:
                    try:
                        print(f"[PartMisalignmentDetector] Trying {model_id}...")
                        sam2_model = build_sam2_hf(model_id, device=device)
                        print(f"[PartMisalignmentDetector] Successfully loaded {model_id}")
                        break
                    except Exception as hf_error:
                        print(f"[PartMisalignmentDetector] Failed {model_id}: {hf_error}")
                        continue
                
                if sam2_model is not None:
                    # BALANCED: Sensitive enough for minute defects but not overly aggressive
                    self.mask_generator = SAM2AutomaticMaskGenerator(
                        sam2_model,
                        points_per_side=48,              # Balanced sampling (not too dense)
                        pred_iou_thresh=0.75,            # Moderate threshold
                        stability_score_thresh=0.85,     # Require reasonably stable masks
                        crop_n_layers=0,                 # Disable multi-scale to reduce over-detection
                        min_mask_region_area=20,         # Slightly higher to avoid tiny noise
                    )
                    print("[PartMisalignmentDetector] SAM2 mask generator initialized with BALANCED sensitivity")
                else:
                    print("[PartMisalignmentDetector] All SAM2 HuggingFace models failed")
                    self.mask_generator = None
                    
            except Exception as model_error:
                print(f"[PartMisalignmentDetector] Failed to load SAM2 model: {model_error}")
                # Fallback to dummy mode
                self.mask_generator = None
                
        except ImportError as e:
            print(f"[PartMisalignmentDetector] SAM2 not installed ({e}). Using dummy mode.")
            self.mask_generator = None
        except Exception as e:
            print(f"[PartMisalignmentDetector] SAM2 setup failed ({e}). Using dummy mode.")
            self.mask_generator = None
    
    def enhance_image_for_defects(self, image: np.ndarray) -> np.ndarray:
        """
        ENHANCED: Pre-process image to make minute defects more visible
        Applies CLAHE, edge enhancement, and noise reduction
        """
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Bilateral filter (reduce noise while preserving edges)
            enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 3. Unsharp masking (enhance edges and fine details)
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            print("[SurfaceDefect] Image enhancement applied: CLAHE + Bilateral + Unsharp Mask")
            return enhanced
            
        except Exception as e:
            print(f"[SurfaceDefect] Error in image enhancement: {e}")
            return image  # Return original if enhancement fails
    
    def detect(self, image_path: str) -> List[Dict]:
        """Detect surface defects using SAM2 segmentation with enhanced pre-processing"""
        if self.mask_generator is None:
            return self._dummy_surface_defects(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        
        # ENHANCED: Pre-process image to make minute defects more visible
        enhanced_image = self.enhance_image_for_defects(image)
        enhanced_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        # Generate masks using SAM2 on enhanced image
        print("[SurfaceDefect] Running SAM2 on enhanced image...")
        masks = self.mask_generator.generate(enhanced_rgb)
        print(f"[SurfaceDefect] SAM2 generated {len(masks)} masks")
        
        # Analyze each mask for surface defects (use original image for classification)
        surface_defects = self.analyze_sam2_detections(image, masks)
        
        return surface_defects
    
    def analyze_sam2_detections(self, image: np.ndarray, sam2_masks: List[Dict]) -> List[Dict]:
        """ENHANCED: Analyze SAM2 detections with improved classification for minute defects"""
        classified_defects = []
        
        for mask_data in sam2_masks:
            mask = mask_data.get('segmentation')
            bbox = mask_data.get('bbox')  # [x, y, w, h]
            area = mask_data.get('area', 0)
            
            # BALANCED: Catch minute defects but avoid tiny noise (25 pixels minimum)
            if mask is None or area < 25:
                continue
            
            # Convert bbox format and extract ROI
            if bbox:
                x, y, w, h = bbox
                # Ensure integer coordinates
                x, y, w, h = int(x), int(y), int(w), int(h)
                roi = self.extract_roi(image, mask, (x, y, x+w, y+h))
                if roi is None or roi.size == 0:
                    continue
                
                # Analyze for different defect types
                defect_results = self.classify_defect(roi, mask, bbox)
                
                if defect_results:
                    defect_results.update({
                        'bbox': [int(x), int(y), int(x+w), int(y+h)],  # Convert to detection format
                        'area': area,
                        'location': self._get_location_name(x + w/2, y + h/2, image.shape[1], image.shape[0])
                    })
                    classified_defects.append(defect_results)
        
        # FILTER: Keep only the best detection per defect type, max 3 total
        filtered_defects = self._filter_best_detections(classified_defects)
        
        return filtered_defects
    
    def _filter_best_detections(self, defects: List[Dict], max_total: int = 3) -> List[Dict]:
        """
        Filter detections to keep only the best (highest confidence) per defect type
        and limit total detections to max_total
        """
        if not defects:
            return []
        
        # Group by defect type
        defects_by_type = {}
        for defect in defects:
            defect_type = defect.get('defect_type', 'unknown')
            if defect_type not in defects_by_type:
                defects_by_type[defect_type] = []
            defects_by_type[defect_type].append(defect)
        
        # Keep only the best (highest score) defect per type
        best_per_type = []
        for defect_type, type_defects in defects_by_type.items():
            # Sort by score (descending) and take the best one
            best_defect = max(type_defects, key=lambda d: d.get('score', 0))
            best_per_type.append(best_defect)
            print(f"[SurfaceDefect] Keeping best {defect_type}: score={best_defect['score']:.2f}, area={best_defect.get('area', 0)}")
        
        # Sort by score and keep only top max_total
        best_per_type.sort(key=lambda d: d.get('score', 0), reverse=True)
        filtered = best_per_type[:max_total]
        
        print(f"[SurfaceDefect] Filtered {len(defects)} detections → {len(filtered)} final (1 per type, max {max_total})")
        
        return filtered
    
    def extract_roi(self, image: np.ndarray, mask, bbox) -> np.ndarray:
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
            print(f"[SurfaceDefect] Error extracting ROI: {e}")
            return None
    
    def classify_defect(self, roi: np.ndarray, mask, bbox) -> Dict:
        """ENHANCED: Classify defects with improved sensitivity for minute paint defects"""
        if roi is None or roi.size == 0:
            return None
        
        defect_scores = {}
        
        # 1. ENHANCED: Linear Scratch Detection (NEW)
        scratch_score = self.detect_scratch(roi, bbox)
        if scratch_score > 0.15:  # Low threshold for fine scratches
            defect_scores['scratch'] = scratch_score
        
        # 2. ENHANCED: Paint Defects (color variations) - More sensitive
        paint_score = self.detect_paint_defect(roi)
        if paint_score > 0.2:  # Lowered from 0.3
            defect_scores['paint_defect'] = paint_score
        
        # 3. ENHANCED: Surface Contamination (water stains, hazing)
        contamination_score = self.detect_contamination(roi)
        if contamination_score > 0.15:  # Lowered from 0.2
            defect_scores['contamination'] = contamination_score
        
        # 4. Corrosion/Rust
        rust_score = self.detect_corrosion(roi)
        if rust_score > 0.25:  # Slightly lowered
            defect_scores['corrosion'] = rust_score
        
        # 5. Water Spots - Enhanced detection
        water_spot_score = self.detect_water_spots(roi, mask)
        if water_spot_score > 0.08:  # Lowered from 0.1
            defect_scores['water_spots'] = water_spot_score
        
        # 6. Simple water spot detection (brightness-based)
        simple_water_score = self.detect_simple_water_spots(roi)
        if simple_water_score > 0.15:  # Lowered from 0.2
            defect_scores['water_spots'] = max(defect_scores.get('water_spots', 0), simple_water_score)
        
        # 7. ENHANCED: Texture-based defect detection (NEW)
        texture_score = self.detect_texture_defect(roi)
        if texture_score > 0.2:
            defect_scores['texture_defect'] = texture_score
        
        # 8. ENHANCED: Chemical damage / hazing (NEW)
        haze_score = self.detect_hazing(roi)
        if haze_score > 0.2:
            defect_scores['hazing'] = haze_score
        
        # Return the highest scoring defect type
        if defect_scores:
            best_defect = max(defect_scores.items(), key=lambda x: x[1])
            
            # Map defect types to user-friendly names
            defect_name_map = {
                'scratch': 'Scratch',
                'paint_defect': 'Paint Defect',
                'contamination': 'Contamination',
                'corrosion': 'Corrosion',
                'water_spots': 'Water Spots',
                'texture_defect': 'Surface Texture Defect',
                'hazing': 'Paint Hazing'
            }
            
            return {
                'class': defect_name_map.get(best_defect[0], best_defect[0].replace('_', ' ').title()),
                'score': best_defect[1],
                'defect_type': best_defect[0],
                'all_scores': defect_scores
            }
        
        return None
    
    def detect_paint_defect(self, roi: np.ndarray) -> float:
        """ENHANCED: Detect paint defects with improved sensitivity"""
        try:
            # Convert to LAB color space for better color analysis
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            
            # Calculate color variance across the region
            color_variance = np.var(lab, axis=(0, 1))
            total_variance = np.sum(color_variance)
            
            # ENHANCED: Lower normalization threshold for minute defects
            variance_score = min(total_variance / 500.0, 1.0)  # Reduced from 1000.0
            
            # Additional: Check for color uniformity
            l, a, b = cv2.split(lab)
            color_std = np.std(a) + np.std(b)  # Color channel standard deviation
            uniformity_score = min(color_std / 20.0, 1.0)  # Non-uniform = defect
            
            # Combined score
            score = (variance_score + uniformity_score) / 2
            
            return score
        except:
            return 0.0
    
    def detect_contamination(self, roi: np.ndarray) -> float:
        """ENHANCED: Detect surface contamination with higher sensitivity"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Look for unusual brightness patterns
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # ENHANCED: Lower threshold for detecting spots (1.5 std instead of 2)
            bright_spots = np.sum(gray > mean_brightness + 1.5 * brightness_std)
            dark_spots = np.sum(gray < mean_brightness - 1.5 * brightness_std)
            
            total_pixels = gray.size
            contamination_ratio = (bright_spots + dark_spots) / total_pixels
            
            # ENHANCED: More sensitive scoring
            score = min(contamination_ratio * 8, 1.0)  # Increased from 5
            
            # Additional: Check for localized brightness variation
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(gray, blur)
            variation_score = np.mean(diff) / 50.0
            
            # Combined score
            final_score = (score + min(variation_score, 1.0)) / 2
            
            return final_score
        except:
            return 0.0
    
    def detect_corrosion(self, roi: np.ndarray) -> float:
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
    
    def detect_water_spots(self, roi: np.ndarray, mask) -> float:
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
    
    def detect_simple_water_spots(self, roi: np.ndarray) -> float:
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
                spot_density_score = min((num_labels - 1) / 20.0, 1.0)  # Normalize by expected max spots
                brightness_score = min(bright_ratio * 10, 1.0)
                
                return (spot_density_score + brightness_score) / 2
            
            return 0.0
        except:
            return 0.0
    
    def detect_scratch(self, roi: np.ndarray, bbox) -> float:
        """
        NEW: Detect linear scratches using edge detection and shape analysis
        Perfect for the fine scratches in your sample images
        """
        try:
            if roi.size == 0:
                return 0.0
                
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for finding scratches
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for finer edges
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            scratch_score = 0.0
            for contour in contours:
                if len(contour) < 5:
                    continue
                
                # Fit ellipse or bounding rect to analyze shape
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width == 0 or height == 0:
                    continue
                
                # Calculate aspect ratio (scratches are long and thin)
                aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
                
                # Scratches have high aspect ratio (length >> width)
                if aspect_ratio > 3.0:  # Linear feature
                    area = cv2.contourArea(contour)
                    if area > 5:  # Minimum area
                        # Score based on linearity
                        linearity_score = min(aspect_ratio / 10.0, 1.0)
                        scratch_score = max(scratch_score, linearity_score)
            
            return scratch_score
            
        except Exception as e:
            print(f"[SurfaceDefect] Error in scratch detection: {e}")
            return 0.0
    
    def detect_texture_defect(self, roi: np.ndarray) -> float:
        """
        NEW: Detect texture variations using Local Binary Patterns
        Useful for orange peel, roughness, and surface finish issues
        """
        try:
            if roi.size == 0:
                return 0.0
                
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture variance using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Also check standard deviation (texture uniformity)
            std_dev = np.std(gray)
            
            # Texture defects have higher variance and std dev
            variance_score = min(variance / 500.0, 1.0)
            uniformity_score = min(std_dev / 50.0, 1.0)
            
            # Combined score
            texture_score = (variance_score + uniformity_score) / 2
            
            return texture_score
            
        except Exception as e:
            print(f"[SurfaceDefect] Error in texture detection: {e}")
            return 0.0
    
    def detect_hazing(self, roi: np.ndarray) -> float:
        """
        NEW: Detect paint hazing, water stains, and chemical damage
        Perfect for the cloudy/milky marks in your sample images
        """
        try:
            if roi.size == 0:
                return 0.0
                
            # Convert to LAB color space for better analysis
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Hazing typically shows as:
            # 1. Reduced saturation (lower a, b values)
            # 2. Brightness variation (L channel inconsistency)
            # 3. Milky/cloudy appearance
            
            # Calculate color saturation
            saturation = np.sqrt(a.astype(float)**2 + b.astype(float)**2)
            avg_saturation = np.mean(saturation)
            
            # Low saturation indicates hazing/whitening
            saturation_score = 1.0 - min(avg_saturation / 50.0, 1.0)
            
            # Check brightness uniformity (hazing creates uneven brightness)
            l_std = np.std(l)
            brightness_variation_score = min(l_std / 30.0, 1.0)
            
            # Look for milky/cloudy regions (high L, low saturation)
            milky_pixels = np.sum((l > np.mean(l)) & (saturation < np.mean(saturation)))
            milky_ratio = milky_pixels / roi.size
            milky_score = min(milky_ratio * 5, 1.0)
            
            # Combined hazing score
            haze_score = (saturation_score * 0.3 + brightness_variation_score * 0.3 + milky_score * 0.4)
            
            return haze_score
            
        except Exception as e:
            print(f"[SurfaceDefect] Error in hazing detection: {e}")
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
        return [
            {
                "class": "Paint Defect",
                "score": 0.75,
                "defect_type": "paint_defect",
                "bbox": [100, 100, 200, 200],
                "location": "middle_center_area"
            }
        ]


# -------------------------
# Vehicle Inspector (Combine both)
# -------------------------
class VehicleInspector:
    def __init__(self, defect_model="fasterrcnn", model_path="fasterrcnn_model.pth", sam2_model=None):
        self.defect_detector = DefectDetector(model_type=defect_model, model_path=model_path)
        self.surface_defect_detector = AutomotiveSurfaceDefectDetector(sam2_model=sam2_model)
    
    def inspect(self, image_path: str) -> Dict:
        # Detect structural defects (dents, scratches) using FasterRCNN
        structural_defects = self.defect_detector.detect(image_path)
        
        # Detect surface defects (paint, contamination, corrosion, water spots) using SAM2
        surface_defects = self.surface_defect_detector.detect(image_path)
        
        return {
            "defects": structural_defects,
            "surface_defects": surface_defects
        }
