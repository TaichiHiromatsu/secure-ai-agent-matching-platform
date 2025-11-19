"""
Submission API endpoints
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db, Submission
from ..services.workflow import run_review_workflow
from pydantic import BaseModel, HttpUrl
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class SubmissionRequest(BaseModel):
    """Submission creation request"""
    agentCardUrl: HttpUrl
    endpointUrl: HttpUrl


class SubmissionResponse(BaseModel):
    """Submission creation response"""
    submissionId: str
    status: str
    message: str


@router.post("/submissions", response_model=SubmissionResponse)
async def create_submission(
    req: SubmissionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new agent submission and start review workflow
    """
    logger.info(f"Creating submission for endpoint: {req.endpointUrl}")
    
    # Check for duplicate endpoint
    existing = db.query(Submission).filter(
        Submission.endpoint_url == str(req.endpointUrl)
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Endpoint already submitted: {existing.id}"
        )
    
    # Create submission
    submission_id = str(uuid.uuid4())
    submission = Submission(
        id=submission_id,
        agent_card_url=str(req.agentCardUrl),
        endpoint_url=str(req.endpointUrl),
        status="precheck_pending"
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    
    logger.info(f"Created submission {submission_id}, starting workflow...")
    
    # Start background review workflow
    background_tasks.add_task(run_review_workflow, submission_id)
    
    return SubmissionResponse(
        submissionId=submission_id,
        status="precheck_pending",
        message="Submission created successfully. Review workflow started."
    )


@router.get("/submissions/{submission_id}")
async def get_submission(submission_id: str, db: Session = Depends(get_db)):
    """
    Get submission details by ID
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    return {
        "submissionId": submission.id,
        "agentCardUrl": submission.agent_card_url,
        "endpointUrl": submission.endpoint_url,
        "status": submission.status,
        "createdAt": submission.created_at.isoformat(),
        "stages": submission.stages
    }


@router.get("/submissions/{submission_id}/progress")
async def get_progress(submission_id: str, db: Session = Depends(get_db)):
    """
    Get review progress for a submission
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    return {
        "submissionId": submission.id,
        "status": submission.status,
        "stages": submission.stages,
        "updatedAt": submission.updated_at.isoformat()
    }


@router.get("/submissions")
async def list_submissions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List all submissions with pagination
    """
    submissions = db.query(Submission).offset(offset).limit(limit).all()
    
    return {
        "submissions": [
            {
                "id": s.id,
                "endpointUrl": s.endpoint_url,
                "status": s.status,
                "createdAt": s.created_at.isoformat()
            }
            for s in submissions
        ],
        "total": db.query(Submission).count()
    }
