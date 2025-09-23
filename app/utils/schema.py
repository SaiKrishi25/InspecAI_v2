import os

def to_response_schema(image_id, detections, visualization_path):
    """Convert defect detections to API response schema"""
    # Normalize path for web route if inside static/outputs
    vis_rel = visualization_path
    if "app/static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    elif "static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    else:
        vis_rel = os.path.basename(visualization_path)
    
    return {
        "image_id": image_id,
        "detections": [
            {
                "bbox": det["bbox"],
                "score": float(det.get("score", 0.0)),
                "defect_type": det.get("class", "unknown"),
                "class_id": det.get("class_id", -1)
            }
            for det in detections
        ],
        "total_defects": len(detections),
        "defect_summary": _get_defect_summary(detections),
        "original_image": f"/uploads/{image_id}.jpg",
        "visualization": f"/outputs/{vis_rel}"
    }

def _get_defect_summary(detections):
    """Generate summary of detected defects"""
    summary = {}
    for det in detections:
        defect_type = det.get("class", "unknown")
        if defect_type not in summary:
            summary[defect_type] = {
                "count": 0,
                "avg_confidence": 0.0,
                "scores": []
            }
        summary[defect_type]["count"] += 1
        summary[defect_type]["scores"].append(det.get("score", 0.0))
    
    # Calculate average confidence for each defect type
    for defect_type in summary:
        scores = summary[defect_type]["scores"]
        summary[defect_type]["avg_confidence"] = sum(scores) / len(scores) if scores else 0.0
        del summary[defect_type]["scores"]  # Remove raw scores from response
    
    return summary

def to_inspection_response_schema(image_id, results, visualization_path):
    """Convert complete inspection results (defects + misalignments) to API response schema"""
    # Normalize path for web route if inside static/outputs
    vis_rel = visualization_path
    if "app/static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    elif "static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    else:
        vis_rel = os.path.basename(visualization_path)
    
    defects = results.get("defects", [])
    surface_defects = results.get("surface_defects", [])
    
    return {
        "image_id": image_id,
        "defects": [
            {
                "bbox": det["bbox"],
                "score": float(det.get("score", 0.0)),
                "defect_type": det.get("class", "unknown"),
                "class_id": det.get("class_id", -1)
            }
            for det in defects
        ],
        "surface_defects": [
            {
                "defect_type": sdef.get("defect_type", "unknown"),
                "class": sdef.get("class", "Unknown"),
                "score": float(sdef.get("score", 0.0)),
                "bbox": sdef.get("bbox", []),
                "location": sdef.get("location", "unknown_area")
            }
            for sdef in surface_defects
        ],
        "total_defects": len(defects),
        "total_surface_defects": len(surface_defects),
        "defect_summary": _get_defect_summary(defects),
        "surface_defect_summary": _get_surface_defect_summary(surface_defects),
        "overall_status": _get_overall_status(defects, surface_defects),
        "original_image": f"/uploads/{image_id}.jpg",
        "visualization": f"/outputs/{vis_rel}"
    }

def _get_surface_defect_summary(surface_defects):
    """Generate summary of detected surface defects"""
    if not surface_defects:
        return {"status": "No surface defects detected"}
    
    defect_types = {}
    high_confidence_count = 0
    
    for sdef in surface_defects:
        defect_type = sdef.get("defect_type", "unknown")
        score = sdef.get("score", 0)
        
        if defect_type not in defect_types:
            defect_types[defect_type] = {"count": 0, "avg_confidence": 0, "scores": []}
        
        defect_types[defect_type]["count"] += 1
        defect_types[defect_type]["scores"].append(score)
        
        if score > 0.7:
            high_confidence_count += 1
    
    # Calculate averages
    for defect_type in defect_types:
        scores = defect_types[defect_type]["scores"]
        defect_types[defect_type]["avg_confidence"] = sum(scores) / len(scores)
        del defect_types[defect_type]["scores"]
    
    return {
        "total_count": len(surface_defects),
        "defect_types": defect_types,
        "high_confidence_count": high_confidence_count,
        "status": "Critical" if high_confidence_count > 2 else "Acceptable"
    }

def _get_overall_status(defects, surface_defects):
    """Get overall inspection status"""
    defect_count = len(defects)
    high_confidence_surface = sum(1 for sdef in surface_defects if sdef.get("score", 0) > 0.7)
    
    if defect_count == 0 and high_confidence_surface == 0:
        return "PASS"
    elif defect_count <= 2 and high_confidence_surface <= 1:
        return "MINOR_ISSUES"
    else:
        return "FAIL"
