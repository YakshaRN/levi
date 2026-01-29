"""
Frontend server for Levitate API.
Run with: uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import os

# Import the main API app
from levitate import app as api_app, s3, S3_BUCKET, S3_COVER_BUCKET, logger

# ---------------- APP ----------------
app = FastAPI(title="Levitate Frontend", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPER FUNCTIONS ----------------
def get_music_id(s3_key: str) -> str:
    """Get a unique ID for a music file (filename without extension)."""
    return os.path.splitext(s3_key)[0]

def get_image_for_music(music_key: str) -> dict | None:
    """Find the latest generated image for a music file."""
    music_id = get_music_id(music_key)
    
    try:
        # List all images in the cover bucket that match this music ID
        response = s3.list_objects_v2(
            Bucket=S3_COVER_BUCKET,
            Prefix=f"{music_id}_"
        )
        
        if "Contents" not in response or len(response["Contents"]) == 0:
            return None
        
        # Sort by last modified to get the latest image
        images = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
        latest_image = images[0]
        
        # Generate presigned URL for the image
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_COVER_BUCKET, 'Key': latest_image["Key"]},
            ExpiresIn=604800  # 7 days
        )
        
        return {
            "key": latest_image["Key"],
            "url": presigned_url,
            "last_modified": latest_image["LastModified"].isoformat()
        }
    except Exception as e:
        logger.exception(f"Failed to get image for music: {music_key}")
        return None

# ---------------- LIST MUSIC ENDPOINT ----------------
@app.get("/music")
def list_music():
    """List all uploaded music files with their associated images."""
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        
        if "Contents" not in response:
            return {"files": []}
        
        files = []
        for obj in response["Contents"]:
            if obj["Key"].lower().endswith(".mp3"):
                music_id = get_music_id(obj["Key"])
                image_info = get_image_for_music(obj["Key"])
                
                files.append({
                    "key": obj["Key"],
                    "id": music_id,
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "has_image": image_info is not None,
                    "image_url": image_info["url"] if image_info else None
                })
        
        # Sort by last modified (newest first)
        files.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return {"files": files}
    except Exception as e:
        logger.exception("Failed to list music files")
        return {"files": [], "error": str(e)}

# ---------------- GET IMAGE FOR MUSIC ----------------
@app.get("/music/image/{s3_key:path}")
def get_music_image(s3_key: str):
    """Get the existing generated image for a music file."""
    image_info = get_image_for_music(s3_key)
    
    if image_info:
        return {"exists": True, **image_info}
    else:
        return {"exists": False}

# ---------------- PLAY MUSIC ENDPOINT ----------------
@app.get("/music/play/{s3_key:path}")
def get_music_url(s3_key: str):
    """Get a presigned URL to play music."""
    try:
        logger.info(f"Getting presigned URL for: {s3_key}")
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600  # 1 hour
        )
        logger.info(f"Generated presigned URL successfully")
        return {"url": presigned_url}
    except Exception as e:
        logger.exception(f"Failed to get music URL for: {s3_key}")
        return {"error": str(e)}

# ---------------- STATIC FILES ----------------
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the API last to avoid conflicts
app.mount("/api", api_app)
