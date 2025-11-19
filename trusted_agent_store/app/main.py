"""
FastAPI single container application for Trusted Agent Store
"""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from .database import init_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trusted Agent Store - Single Container",
    description="Unified trusted agent submission and review platform",
    version="1.0.0"
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Trusted Agent Store application...")
    init_db()
    logger.info("Database initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trusted-agent-store"}


@app.get("/")
async def root(request: Request):
    """Root endpoint - home page"""
    return templates.TemplateResponse("index.html", {"request": request})


# Import routers after app creation to avoid circular imports
from .routers import submissions, reviews, ui

app.include_router(submissions.router, prefix="/api", tags=["submissions"])
app.include_router(reviews.router, prefix="/api/review", tags=["reviews"])
app.include_router(ui.router, tags=["ui"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
