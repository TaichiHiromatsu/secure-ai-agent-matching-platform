from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
import uuid

router = APIRouter(
    prefix="/api/reviews",
    tags=["reviews"],
)

@router.post("/{submission_id}/decision", response_model=schemas.Submission)
def submit_review_decision(
    submission_id: str,
    review: schemas.ReviewRequest,
    db: Session = Depends(get_db)
):
    submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    if review.action == schemas.ReviewAction.APPROVE:
        submission.state = "approved"
    elif review.action == schemas.ReviewAction.REJECT:
        submission.state = "rejected"

    # Log the decision (simplified logic)
    # In a real app, we would create a TrustScoreHistory entry here

    db.commit()
    db.refresh(submission)
    return submission

@router.post("/{submission_id}/score", response_model=schemas.Submission)
def update_trust_score(
    submission_id: str,
    score_update: schemas.TrustScoreUpdate,
    db: Session = Depends(get_db)
):
    submission = db.query(models.Submission).filter(models.Submission.id == submission_id).first()
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    if score_update.security_score is not None:
        submission.security_score = score_update.security_score
    if score_update.functional_score is not None:
        submission.functional_score = score_update.functional_score
    if score_update.judge_score is not None:
        submission.judge_score = score_update.judge_score
    if score_update.implementation_score is not None:
        submission.implementation_score = score_update.implementation_score

    # Recalculate total score (simplified)
    submission.trust_score = (
        submission.security_score +
        submission.functional_score +
        submission.judge_score +
        submission.implementation_score
    )

    if score_update.reasoning:
        submission.score_breakdown = score_update.reasoning

    db.commit()
    db.refresh(submission)
    return submission
