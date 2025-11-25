from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from .. import models
from ..database import get_db

router = APIRouter(
    tags=["ui"],
)

templates = Jinja2Templates(directory="app/templates")

@router.get("/submit", response_class=HTMLResponse)
async def submit_page(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    submissions = db.query(models.Submission).order_by(models.Submission.created_at.desc()).all()
    return templates.TemplateResponse("admin/dashboard.html", {"request": request, "submissions": submissions})

@router.get("/admin/review/{submission_id}", response_class=HTMLResponse)
async def admin_review(request: Request, submission_id: str, db: Session = Depends(get_db)):
    submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return templates.TemplateResponse("admin/review.html", {"request": request, "submission": submission})

@router.get("/submissions/{submission_id}/status", response_class=HTMLResponse)
async def submission_status(request: Request, submission_id: str, db: Session = Depends(get_db)):
    submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return templates.TemplateResponse("status.html", {"request": request, "submission": submission})
