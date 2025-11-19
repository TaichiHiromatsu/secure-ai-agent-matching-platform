"""
UI routing endpoints
"""
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from ..database import get_db, Submission

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/submit")
async def submit_page(request: Request):
    """Agent submission form page"""
    return templates.TemplateResponse("submit.html", {"request": request})


@router.get("/review/{submission_id}")
async def review_page(
    request: Request,
    submission_id: str,
    db: Session = Depends(get_db)
):
    """Review status page for a specific submission"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    
    if not submission:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Submission not found"},
            status_code=404
        )
    
    return templates.TemplateResponse("review.html", {
        "request": request,
        "submission": submission
    })


@router.get("/submissions")
async def submissions_list_page(
    request: Request,
    db: Session = Depends(get_db)
):
    """List all submissions page"""
    submissions = db.query(Submission).order_by(Submission.created_at.desc()).all()
    
    return templates.TemplateResponse("submissions_list.html", {
        "request": request,
        "submissions": submissions
    })
