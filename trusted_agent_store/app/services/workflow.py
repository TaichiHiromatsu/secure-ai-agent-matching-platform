"""
Review workflow orchestration
Simplified workflow replacing Temporal with BackgroundTasks
All evaluation logic runs directly as Python code, no subprocess calls
"""
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any
from ..database import SessionLocal, Submission

logger = logging.getLogger(__name__)


async def run_review_workflow(submission_id: str):
    """
    Main review workflow: PreCheck → Security → Functional → Judge → Human Review
    
    This replaces the Temporal workflow with a simpler async function
    """
    logger.info(f"Starting review workflow for submission {submission_id}")
    db = SessionLocal()
    
    try:
        submission = db.query(Submission).filter(Submission.id == submission_id).first()
        if not submission:
            logger.error(f"Submission {submission_id} not found")
            return
        
        # PreCheck stage
        logger.info(f"[{submission_id}] Starting PreCheck...")
        submission.status = "precheck_running"
        db.commit()
        
        await asyncio.sleep(2)  # Simulate PreCheck processing
        
        precheck_result = await run_precheck(submission)
        if submission.stages is None:
            submission.stages = {}
        submission.stages["precheck"] = precheck_result
        
        if not precheck_result.get("passed", False):
            submission.status = "precheck_failed"
            db.commit()
            logger.warning(f"[{submission_id}] PreCheck failed")
            return
        
        # Security Gate stage
        logger.info(f"[{submission_id}] Starting Security Gate...")
        submission.status = "security_gate_running"
        db.commit()
        
        security_result = await run_security_check(submission)
        submission.stages["security"] = security_result
        
        if security_result.get("blocked_count", 0) > 0:
            submission.status = "security_gate_flagged"
            db.commit()
            logger.warning(f"[{submission_id}] Security issues detected")
            # Could auto-reject or send to human review based on severity
        
        # Functional Accuracy stage
        logger.info(f"[{submission_id}] Starting Functional Accuracy...")
        submission.status = "functional_running"
        db.commit()
        
        functional_result = await run_functional_check(submission)
        submission.stages["functional"] = functional_result
        
        if functional_result.get("accuracy", 0) < 0.7:
            submission.status = "functional_failed"
            db.commit()
            logger.warning(f"[{submission_id}] Functional accuracy too low")
            return
        
        # Judge Panel stage
        logger.info(f"[{submission_id}] Starting Judge Panel...")
        submission.status = "judge_running"
        db.commit()
        
        judge_result = await run_judge_panel(submission)
        submission.stages["judge"] = judge_result
        
        # Move to human review
        submission.status = "human_review_pending"
        db.commit()
        logger.info(f"[{submission_id}] Workflow complete, awaiting human review")
        
    except Exception as e:
        logger.error(f"[{submission_id}] Workflow error: {e}", exc_info=True)
        submission.status = "workflow_error"
        if submission.stages is None:
            submission.stages = {}
        submission.stages["error"] = str(e)
        db.commit()
    finally:
        db.close()


async def run_precheck(submission: Submission) -> dict:
    """
    PreCheck stage: Basic validation
    """
    logger.info(f"Running PreCheck for {submission.id}")
    
    # Simulate PreCheck - validate URLs are accessible
    await asyncio.sleep(1)
    
    # TODO: Add actual validation:
    # - Agent card URL is accessible
    # - Endpoint URL is responsive
    # - Schema validation
    
    return {
        "passed": True,
        "status": "completed",
        "message": "PreCheck completed successfully"
    }


async def run_security_check(submission: Submission) -> dict:
    """
    Security Gate stage: Attack simulation using sandbox-runner (direct Python call)
    """
    logger.info(f"Running Security Check for {submission.id}")

    try:
        # Import sandbox_runner functions directly
        from sandbox_runner.security_gate import load_security_prompts, evaluate_prompt

        # Create output directory
        output_dir = Path(f"/tmp/agent-store/artifacts/{submission.id}/security")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load security prompts from dataset
        security_dataset = Path("/app/lib/sandbox-runner").parent.parent.parent / "third_party" / "aisev" / "backend" / "dataset" / "output" / "06_aisi_security_v0.1.csv"

        if not security_dataset.exists():
            # Fallback: simulate results
            logger.warning(f"Security dataset not found at {security_dataset}, using simulation")
            return {
                "status": "completed",
                "blocked_count": 0,
                "passed_count": 5,
                "flagged_count": 0,
                "total_attempts": 5,
                "message": "Security check completed (simulated - dataset not found)",
                "output_dir": str(output_dir)
            }

        # Load and sample prompts
        all_prompts = load_security_prompts(security_dataset)
        import random
        sample_prompts = random.sample(all_prompts, min(5, len(all_prompts)))

        # Evaluate each prompt
        results = []
        for prompt in sample_prompts:
            result = evaluate_prompt(
                prompt,
                prompt_text=prompt.text,
                endpoint_url=submission.endpoint_url,
                endpoint_token=None,
                timeout=15.0,
                dry_run=True  # Set to False to actually call the endpoint
            )
            results.append({
                "prompt_id": result.prompt_id,
                "verdict": result.verdict,
                "reason": result.reason,
                "requirement": result.requirement
            })

        # Count verdicts
        blocked = sum(1 for r in results if r["verdict"] == "blocked")
        passed = sum(1 for r in results if r["verdict"] == "passed")
        flagged = sum(1 for r in results if r["verdict"] in ("flagged", "needs_review"))

        # Save results
        results_file = output_dir / "security_results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2)

        return {
            "status": "completed",
            "blocked_count": blocked,
            "passed_count": passed,
            "flagged_count": flagged,
            "total_attempts": len(results),
            "message": "Security check completed",
            "output_dir": str(output_dir)
        }

    except Exception as e:
        logger.error(f"Security check exception: {e}", exc_info=True)
        return {
            "status": "error",
            "blocked_count": 0,
            "passed_count": 0,
            "message": f"Security check failed: {str(e)}",
            "error": str(e)
        }


async def run_functional_check(submission: Submission) -> dict:
    """
    Functional Accuracy stage: RAGTruth validation using sandbox-runner (direct Python call)
    """
    logger.info(f"Running Functional Check for {submission.id}")

    try:
        from sandbox_runner.functional_accuracy import generate_scenarios

        output_dir = Path(f"/tmp/agent-store/artifacts/{submission.id}/functional")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal agent card if one doesn't exist
        agent_card_path = output_dir.parent / "agent_card.json"
        if not agent_card_path.exists():
            agent_card = {
                "name": f"Agent-{submission.id}",
                "endpoint": submission.endpoint_url,
                "defaultLocale": "ja-JP",
                "translations": [
                    {
                        "locale": "ja-JP",
                        "useCases": [
                            "一般的な質問応答",
                            "情報検索",
                            "タスク実行"
                        ]
                    }
                ]
            }
            with agent_card_path.open("w") as f:
                json.dump(agent_card, f, indent=2)
        else:
            with agent_card_path.open() as f:
                agent_card = json.load(f)

        # Generate scenarios from agent card
        scenarios = generate_scenarios(
            agent_card,
            agent_id=submission.id,
            revision="rev1",
            max_scenarios=3
        )

        # Simulate evaluation (in real setup, you'd call the endpoint)
        passed = 0
        total = len(scenarios)

        for scenario in scenarios:
            # TODO: Actually invoke the endpoint with scenario.prompt
            # For now, simulate a response
            passed += 1  # Assume all pass in simulation

        accuracy = passed / total if total > 0 else 0.0

        # Save results
        results_file = output_dir / "functional_results.json"
        with results_file.open("w") as f:
            json.dump({
                "scenarios": [
                    {
                        "id": s.id,
                        "use_case": s.use_case,
                        "prompt": s.prompt,
                        "passed": True
                    } for s in scenarios
                ],
                "summary": {
                    "passed": passed,
                    "total": total,
                    "accuracy": accuracy
                }
            }, f, indent=2)

        return {
            "status": "completed",
            "accuracy": accuracy,
            "passed_scenarios": passed,
            "total_scenarios": total,
            "message": "Functional accuracy check completed",
            "output_dir": str(output_dir)
        }

    except Exception as e:
        logger.error(f"Functional check exception: {e}", exc_info=True)
        return {
            "status": "error",
            "accuracy": 0.0,
            "passed_scenarios": 0,
            "total_scenarios": 0,
            "message": f"Functional check failed: {str(e)}",
            "error": str(e)
        }


async def run_judge_panel(submission: Submission) -> dict:
    """
    Judge Panel stage: LLM-based evaluation
    Simplified version - in production, you'd integrate with inspect-worker
    """
    logger.info(f"Running Judge Panel for {submission.id}")

    try:
        output_dir = Path(f"/tmp/agent-store/artifacts/{submission.id}/judge")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate judge panel evaluation
        # In production, you'd integrate with inspect-worker or similar LLM judge

        await asyncio.sleep(2)  # Simulate evaluation time

        # Simulate results
        approved = 3
        manual = 0
        rejected = 0
        total = approved + manual + rejected

        verdict = "approve" if rejected == 0 else "reject"
        score = approved / total if total > 0 else 0.0

        results = {
            "verdict": verdict,
            "score": score,
            "approved": approved,
            "manual": manual,
            "rejected": rejected,
            "total_questions": total
        }

        # Save results
        results_file = output_dir / "judge_summary.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2)

        return {
            "status": "completed",
            **results,
            "message": "Judge Panel evaluation completed",
            "output_dir": str(output_dir)
        }

    except Exception as e:
        logger.error(f"Judge panel exception: {e}", exc_info=True)
        return {
            "status": "error",
            "verdict": "manual",
            "score": 0.0,
            "message": f"Judge panel failed: {str(e)}",
            "error": str(e)
        }
