
import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from app.detector.model import VehicleInspector
from app.utils.visualize import draw_inspection_results
from app.utils.schema import to_inspection_response_schema

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "app/static/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "app/static/outputs")

print("Defect Detection Pipeline initialized")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize vehicle inspector (includes defect detection + surface defect detection)
inspector = VehicleInspector(
    defect_model=os.environ.get("MODEL_TYPE", "yolo"),  # "yolo" or "fasterrcnn"
    model_path=os.environ.get("MODEL_PATH", "best.pt"),
    sam2_model=os.environ.get("SAM2_MODEL", None)  # SAM2 model from HuggingFace
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/infer", methods=["POST"])
def infer():
    # Accept file upload under 'image'
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided under form field 'image'."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    # Get detection mode (default to full inspection)
    detection_mode = request.form.get('detection_mode', 'yolo_sam2')
    
    image_id = str(uuid.uuid4())[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    out_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
    file.save(in_path)

    if detection_mode == 'yolo_only':
        # YOLO only - just defect detection
        defects = inspector.defect_detector.detect(in_path)
        inspection_results = {
            "defects": defects,
            "misalignments": []  # No misalignments in YOLO-only mode
        }
        print(f"[API] YOLO-only mode: Found {len(defects)} defects")
    else:
        # Full Vehicle Inspection (defects + misalignments)
        inspection_results = inspector.inspect(in_path)
        print(f"[API] Full inspection mode: Found {len(inspection_results.get('defects', []))} structural defects, {len(inspection_results.get('surface_defects', []))} surface defects")
    
    # Visualization
    draw_inspection_results(in_path, inspection_results, out_path)
    # JSON response
    response = to_inspection_response_schema(image_id=image_id, results=inspection_results, visualization_path=out_path)
    return jsonify(response)

@app.route("/outputs/<filename>")
def outputs(filename):
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"[DEBUG] Serving output file: {filename} from {abs_output_dir}")
    file_path = os.path.join(abs_output_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_output_dir, filename, as_attachment=False)

@app.route("/uploads/<filename>")
def uploads(filename):
    abs_upload_dir = os.path.abspath(UPLOAD_DIR)
    print(f"[DEBUG] Serving upload file: {filename} from {abs_upload_dir}")
    file_path = os.path.join(abs_upload_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_upload_dir, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
