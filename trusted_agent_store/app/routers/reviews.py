"""
Review API endpoints for Human Review
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db, Submission
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class ReviewDecision(BaseModel):
    """Human review decision"""
    decision: str  # "approve" or "reject"
    reason: str
    reviewerNote: str | None = None


@router.post("/{submission_id}/approve")
async def approve_submission(
    submission_id: str,
    db: Session = Depends(get_db)
):
    """
    Approve a submission and publish the agent
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    logger.info(f"Approving submission {submission_id}")
    
    submission.status = "approved"
    db.commit()
    
    return {
        "submissionId": submission_id,
        "status": "approved",
        "message": "Submission approved and published"
    }


@router.post("/{submission_id}/reject")
async def reject_submission(
    submission_id: str,
    reason: str = "Manual rejection",
    db: Session = Depends(get_db)
):
    """
    Reject a submission
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    logger.info(f"Rejecting submission {submission_id}: {reason}")
    
    submission.status = "rejected"
    if submission.stages is None:
        submission.stages = {}
    submission.stages["rejection_reason"] = reason
    db.commit()
    
    return {
        "submissionId": submission_id,
        "status": "rejected",
        "message": f"Submission rejected: {reason}"
    }


@router.post("/{submission_id}/decision")
async def submit_decision(
    submission_id: str,
    decision: ReviewDecision,
    db: Session = Depends(get_db)
):
    """
    Submit a human review decision (approve/reject)
    """
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if decision.decision == "approve":
        submission.status = "approved"
        logger.info(f"Approved submission {submission_id}: {decision.reason}")
    elif decision.decision == "reject":
        submission.status = "rejected"
        logger.info(f"Rejected submission {submission_id}: {decision.reason}")
    else:
        raise HTTPException(status_code=400, detail="Invalid decision")
    
    if submission.stages is None:
        submission.stages = {}
    submission.stages["human_review"] = {
        "decision": decision.decision,
        "reason": decision.reason,
        "reviewerNote": decision.reviewerNote
    }
    db.commit()
    
    return {
        "submissionId": submission_id,
        "status": submission.status,
        "message": f"Submission {decision.decision}d"
    }
