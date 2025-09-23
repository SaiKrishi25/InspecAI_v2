#!/usr/bin/env python3
"""
Simple example of how to use the SAM2 Surface Defect Detector
============================================================

This is a minimal example showing how to use the detector programmatically.
For Google Colab usage, see the SAM2_Surface_Defect_Demo.ipynb notebook.

Usage:
    python example_usage.py path/to/your/image.jpg
"""

import sys
import os
from sam2_surface_defect_detector import SAM2SurfaceDefectDetector, visualize_results, print_detection_summary

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python example_usage.py <image_path>")
        print("Example: python example_usage.py car_surface.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print("🔍 SAM2 Surface Defect Detection Example")
    print("=" * 50)
    print(f"Input image: {image_path}")
    print()
    
    # Initialize detector
    print("Initializing SAM2 Surface Defect Detector...")
    detector = SAM2SurfaceDefectDetector(
        sam2_model="facebook/sam2-hiera-tiny",  # Fast model for demo
        device=None,  # Auto-detect CUDA/CPU
        max_detections=1  # Return only the most significant defect
    )
    
    # Run detection
    print("\nRunning surface defect detection...")
    detections = detector.detect_surface_defects(image_path)
    
    # Print results
    print_detection_summary(detections)
    
    # Create visualization
    if detections:
        print("\nCreating visualization...")
        output_path = f"detected_defects_{os.path.basename(image_path)}"
        visualize_results(image_path, detections, save_path=output_path)
        print(f"✅ Results saved to: {output_path}")
    else:
        print("No defects detected - no visualization created.")
    
    print("\n🎉 Detection completed!")

if __name__ == "__main__":
    main()
