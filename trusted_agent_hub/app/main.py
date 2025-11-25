from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .database import engine, Base
from .routers import submissions, reviews, ui, agents
from app.services.agent_registry import load_agents
import os

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Trusted Agent Hub")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(submissions.router)
app.include_router(reviews.router)
app.include_router(ui.router)
app.include_router(agents.router)
app.include_router(agents.api_router)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    agents_list = load_agents()
    # Sort by updated_at (or created_at) descending (newest first)
    agents_sorted = sorted(
        agents_list,
        key=lambda a: a.updated_at or a.created_at or "",
        reverse=True
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "agents": agents_sorted,
        },
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}
