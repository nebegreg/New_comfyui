"""
REST API for Mountain Studio
=============================

FastAPI server for automation and remote generation.

Author: Mountain Studio Pro Team
"""

import logging
from typing import Optional, Dict
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available")


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Mountain Studio API", version="1.0.0")

    class TerrainRequest(BaseModel):
        width: int = 512
        height: int = 512
        preset: Optional[str] = None

    jobs: Dict[str, Dict] = {}

    @app.post("/generate/terrain")
    async def generate_terrain(request: TerrainRequest):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'status': 'started', 'type': 'terrain'}
        return {'job_id': job_id}

    @app.get("/status/{job_id}")
    async def get_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404)
        return jobs[job_id]

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    if not FASTAPI_AVAILABLE:
        return
    import uvicorn
    uvicorn.run(app, host=host, port=port)
