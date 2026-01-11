# InspecAI - AI-Powered Vehicle Inspection System

AI-powered vehicle surface and structural defect detection system using FasterRCNN and SAM2 models with real-time analytics, Gemini AI assistant, and cloud storage integration.

## Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core runtime |
| Flask | 3.0.3 | REST API framework |
| PyTorch | 2.2+ | Deep learning framework |
| FasterRCNN | ResNet50-FPN | Structural defect detection |
| SAM2 | HuggingFace | Surface defect segmentation |
| OpenCV | 4.10.0 | Image processing & camera |
| SQLite | 3 | Local database |
| ReportLab | 4.0+ | PDF report generation |
| Google Generative AI | 0.3+ | Gemini AI integration |
| Google Cloud Storage | 2.14+ | Cloud file storage |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2 | UI framework |
| TypeScript | 5.3 | Type safety |
| Material-UI | 5.15 | Component library |
| Vite | 5.0 | Build tool |
| Recharts | 2.10 | Data visualization |
| Axios | 1.6 | HTTP client |
| React Router | 6.21 | Navigation |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         React Frontend (Port 3000)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │
│  │Dashboard │  │InspecInfer│ │ Reports  │  │DigitalAssistant  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ REST API
┌───────────────────────────┴─────────────────────────────────────────┐
│                       Flask Backend (Port 8000)                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  VehicleInspector│  │  GeminiService   │  │   GCSService     │   │
│  │  ├─FasterRCNN    │  │  (AI Assistant)  │  │ (Cloud Storage)  │   │
│  │  └─SAM2          │  └──────────────────┘  └──────────────────┘   │
│  └──────────────────┘                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ ReportGenerator  │  │  DetectionStore  │  │   ReportStore    │   │
│  │ (PDF + Gemini)   │  │   (SQLite)       │  │    (SQLite)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Detection Models

### FasterRCNN (Structural Defects)
- **Architecture**: ResNet50 + Feature Pyramid Network (FPN)
- **Classes**: Dent, Scratch
- **Speed**: ~2-3 seconds
- **Confidence Threshold**: 0.5

### SAM2 (Surface Defects)
- **Model**: `facebook/sam2-hiera-tiny` (HuggingFace)
- **Classes**: Paint Defect, Water Spots, Hazing, Contamination, Corrosion, Texture Defect
- **Speed**: ~5-8 seconds
- **Image Enhancement**: CLAHE + Bilateral Filter + Unsharp Masking

## Project Structure

```
InspecAI_v2/
├── app/
│   ├── main.py              # Flask app & API endpoints
│   ├── database.py          # SQLite operations (DetectionStore, ReportStore)
│   ├── gemini_service.py    # Gemini AI chat integration
│   ├── gcs_service.py       # Google Cloud Storage integration
│   ├── report_generator.py  # PDF report generation with AI analysis
│   ├── prompt_loader.py     # YAML prompt management
│   ├── prompts.yaml         # Centralized AI prompts configuration
│   ├── detector/
│   │   └── model.py         # FasterRCNN + SAM2 detection models
│   ├── utils/
│   │   ├── visualize.py     # Detection visualization
│   │   └── schema.py        # Data validation
│   ├── static/              # uploads/, outputs/, reports/
│   └── data/inspecai.db     # SQLite database
├── frontend/
│   ├── src/
│   │   ├── pages/           # Dashboard, InspecInfer, Reports
│   │   ├── components/      # Layout, Dashboard widgets, DigitalAssistant
│   │   ├── services/        # api.ts, gemini.ts
│   │   └── theme.ts         # MUI theme configuration
│   └── vite.config.ts
├── fasterrcnn_model.pth     # Trained FasterRCNN weights
├── requirements.txt         # Python dependencies
└── start_react_ui.bat       # Startup script
```

## API Endpoints

### Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/capture` | Capture & analyze image |
| GET | `/video_feed` | MJPEG camera stream |
| GET | `/api/camera/status` | Camera availability |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/overview` | Dashboard statistics |
| GET | `/api/analytics/time-series` | Detection trends |
| GET | `/api/analytics/defect-distribution` | Defect type breakdown |
| GET | `/api/detections` | Recent detections |
| GET | `/api/detections/{id}` | Single detection details |

### Reports
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/reports/list` | Paginated report list |
| GET | `/api/reports/search` | Search reports |
| GET | `/api/reports/{id}` | Single report |
| GET | `/reports/{filename}` | PDF download |

### AI Assistant
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Gemini AI chat |
| GET | `/api/chat/suggestions` | Page-specific suggestions |

## Database Schema

```sql
-- Detections
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    detection_id TEXT UNIQUE,
    timestamp DATETIME,
    image_id TEXT,
    total_defects INTEGER,
    detection_mode TEXT,  -- 'fasterrcnn_only' or 'sam2_only'
    report_id TEXT
);

-- Defects
CREATE TABLE defects (
    id INTEGER PRIMARY KEY,
    detection_id TEXT,
    defect_id TEXT,
    defect_type TEXT,     -- 'Dent', 'Scratch', 'Paint Defect', etc.
    confidence_score REAL,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 INTEGER,
    location TEXT
);

-- Reports
CREATE TABLE reports (
    id INTEGER PRIMARY KEY,
    report_id TEXT UNIQUE,
    detection_id TEXT,
    pdf_path TEXT,        -- Local path or GCS URI
    json_data TEXT,
    created_at DATETIME
);
```

## Installation

```bash
# 1. Clone and install backend dependencies
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Start application
start_react_ui.bat  # Windows
# OR manually:
# Terminal 1: python -m app.main
# Terminal 2: cd frontend && npm run dev
```

## Configuration

### Environment Variables
```bash
# .env
GEMINI_API_KEY=your_gemini_api_key
GCS_CREDENTIALS_PATH=inspec-ai_storage.json
GCS_PROJECT_ID=your_project_id
GCS_REPORTS_BUCKET=inspecai-reports
```

### Detection Sensitivity (app/detector/model.py)
```python
# SAM2 thresholds
defect_types = {
    'paint_defect': {'color_variance_threshold': 300, 'min_area': 15},
    'water_spots': {'circularity_threshold': 0.5, 'min_area': 8},
    'scratch': {'min_length': 10, 'max_width': 5},
}

# SAM2 mask generator
SAM2AutomaticMaskGenerator(
    points_per_side=48,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.85,
    min_mask_region_area=20,
)
```

## Features

### Digital Assistant (Gemini AI)
- Context-aware chat based on current page
- Live database query for detection statistics
- YAML-based prompt configuration (`prompts.yaml`)
- Suggested questions per page

### Cloud Storage (Google Cloud Storage)
- Automatic PDF report upload
- Signed URL generation (60-min expiration)
- Fallback to local storage if GCS unavailable

### Report Generation
- AI-powered defect analysis using Gemini 2.0 Flash
- Template fallbacks when AI unavailable
- PDF with images, defect table, and recommendations

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| Node.js | 18+ | 20+ |
| RAM | 4GB | 8GB+ |
| GPU | Optional | CUDA-enabled (speeds up SAM2) |
| Camera | USB/Built-in | 640x480+ |

## Quick Reference

| Access Point | URL |
|--------------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Camera Feed | http://localhost:8000/video_feed |

## License

All rights reserved. Developed for vehicle surface inspection applications.
