"""
Job CLI commands for Kernle Commerce.

Provides command-line interface for job marketplace operations:
- Client commands: create, list, show, fund, applications, accept, approve, cancel, dispute
- Worker commands: search, apply, deliver
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


logger = logging.getLogger(__name__)


def _get_job_service():
    """Get a job service instance.

    For now, returns an InMemory storage-backed service.
    In production, this will use Supabase storage.
    """
    from kernle.commerce.jobs.service import JobService
    from kernle.commerce.jobs.storage import InMemoryJobStorage

    storage = InMemoryJobStorage()
    return JobService(storage)


def _parse_deadline(deadline_str: str) -> datetime:
    """Parse deadline string into datetime.

    Accepts:
    - ISO format: 2026-02-15T00:00:00
    - Relative: 1d, 7d, 2w, 1m (days, weeks, months)
    - Date only: 2026-02-15
    """
    now = datetime.now(timezone.utc)

    # Relative format
    if deadline_str.endswith("d"):
        days = int(deadline_str[:-1])
        return now + timedelta(days=days)
    elif deadline_str.endswith("w"):
        weeks = int(deadline_str[:-1])
        return now + timedelta(weeks=weeks)
    elif deadline_str.endswith("m"):
        months = int(deadline_str[:-1])
        return now + timedelta(days=months * 30)  # Approximate

    # ISO or date-only format
    try:
        # Try full ISO format
        if "T" in deadline_str:
            dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        else:
            # Date only - set to end of day
            dt = datetime.strptime(deadline_str, "%Y-%m-%d")
            return dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(
            f"Invalid deadline format: {deadline_str}. "
            "Use ISO format (2026-02-15T00:00:00), date (2026-02-15), "
            "or relative (1d, 7d, 2w, 1m)"
        ) from e


def cmd_job(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle job subcommands."""
    action = args.job_action

    handlers = {
        # Client commands
        "create": _job_create,
        "list": _job_list,
        "show": _job_show,
        "fund": _job_fund,
        "applications": _job_applications,
        "accept": _job_accept,
        "approve": _job_approve,
        "cancel": _job_cancel,
        "dispute": _job_dispute,
        # Worker commands
        "search": _job_search,
        "apply": _job_apply,
        "deliver": _job_deliver,
    }

    handler = handlers.get(action)
    if handler:
        handler(args, k)
    else:
        print(f"Unknown job action: {action}")
        print(
            "Available actions: create, list, show, fund, applications, accept, approve, cancel, dispute, search, apply, deliver"
        )


# =============================================================================
# Client Commands
# =============================================================================


def _job_create(args: "argparse.Namespace", k: "Kernle") -> None:
    """Create a new job listing."""
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)

    try:
        # Parse deadline
        deadline = _parse_deadline(args.deadline)

        # Collect skills
        skills = args.skill or []

        service = _get_job_service()
        job = service.create_job(
            client_id=agent_id,
            title=args.title,
            description=args.description or f"Job: {args.title}",
            budget_usdc=float(args.budget),
            deadline=deadline,
            skills_required=skills,
        )

        if output_json:
            print(json.dumps(job.to_dict(), indent=2, default=str))
        else:
            print("✅ Job created successfully!")
            print("")
            print(f"  ID:       {job.id}")
            print(f"  Title:    {job.title}")
            print(f"  Budget:   ${job.budget_usdc:.2f} USDC")
            print(f"  Deadline: {job.deadline.strftime('%Y-%m-%d %H:%M UTC')}")
            if skills:
                print(f"  Skills:   {', '.join(skills)}")
            print("")
            print("Next steps:")
            print(f"  1. Fund the job: kernle job fund {job.id}")
            print(f"  2. View applications: kernle job applications {job.id}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error creating job: {e}")


def _job_list(args: "argparse.Namespace", k: "Kernle") -> None:
    """List jobs."""
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)
    mine = getattr(args, "mine", False)
    status_filter = getattr(args, "status", None)
    limit = getattr(args, "limit", 20)

    try:
        from kernle.commerce.jobs.models import JobStatus
        from kernle.commerce.jobs.service import JobSearchFilters

        service = _get_job_service()

        filters = JobSearchFilters(limit=limit)
        if mine:
            filters.client_id = agent_id
        if status_filter:
            filters.status = JobStatus(status_filter)

        jobs = service.list_jobs(filters)

        if output_json:
            result = [job.to_dict() for job in jobs]
            print(json.dumps(result, indent=2, default=str))
        else:
            if not jobs:
                print("No jobs found.")
                if not mine:
                    print("\nTip: Use 'kernle job search' to find available jobs.")
                return

            title = "My Jobs" if mine else "Jobs"
            print(f"📋 {title} ({len(jobs)} found)")
            print("=" * 60)

            for job in jobs:
                status_emoji = {
                    "open": "🟡",
                    "funded": "💰",
                    "accepted": "🔵",
                    "delivered": "📦",
                    "completed": "✅",
                    "disputed": "⚠️",
                    "cancelled": "❌",
                }.get(job.status, "⚪")

                deadline_str = job.deadline.strftime("%Y-%m-%d") if job.deadline else "N/A"
                print(f"\n  {status_emoji} {job.title}")
                print(f"     ID: {job.id[:8]}... | ${job.budget_usdc:.0f} | Due: {deadline_str}")
                if job.skills_required:
                    print(f"     Skills: {', '.join(job.skills_required)}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error listing jobs: {e}")


def _job_show(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show job details."""
    job_id = args.job_id
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job = service.get_job(job_id)

        if output_json:
            print(json.dumps(job.to_dict(), indent=2, default=str))
        else:
            status_emoji = {
                "open": "🟡 Open",
                "funded": "💰 Funded",
                "accepted": "🔵 In Progress",
                "delivered": "📦 Delivered",
                "completed": "✅ Completed",
                "disputed": "⚠️ Disputed",
                "cancelled": "❌ Cancelled",
            }.get(job.status, job.status)

            print("📄 Job Details")
            print("=" * 60)
            print(f"  Title:       {job.title}")
            print(f"  ID:          {job.id}")
            print(f"  Status:      {status_emoji}")
            print(f"  Budget:      ${job.budget_usdc:.2f} USDC")
            print(f"  Client:      {job.client_id}")
            if job.worker_id:
                print(f"  Worker:      {job.worker_id}")
            print(
                f"  Deadline:    {job.deadline.strftime('%Y-%m-%d %H:%M UTC') if job.deadline else 'N/A'}"
            )

            if job.skills_required:
                print(f"  Skills:      {', '.join(job.skills_required)}")

            print("")
            print("Description:")
            print(f"  {job.description}")

            if job.escrow_address:
                print("")
                print(f"Escrow:        {job.escrow_address}")

            if job.deliverable_url:
                print("")
                print(f"Deliverable:   {job.deliverable_url}")
                if job.deliverable_hash:
                    print(f"Hash:          {job.deliverable_hash}")

            # Timestamps
            print("")
            print("Timeline:")
            if job.created_at:
                print(f"  Created:     {job.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
            if job.funded_at:
                print(f"  Funded:      {job.funded_at.strftime('%Y-%m-%d %H:%M UTC')}")
            if job.accepted_at:
                print(f"  Accepted:    {job.accepted_at.strftime('%Y-%m-%d %H:%M UTC')}")
            if job.delivered_at:
                print(f"  Delivered:   {job.delivered_at.strftime('%Y-%m-%d %H:%M UTC')}")
            if job.completed_at:
                print(f"  Completed:   {job.completed_at.strftime('%Y-%m-%d %H:%M UTC')}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_fund(args: "argparse.Namespace", k: "Kernle") -> None:
    """Fund a job (deploy escrow)."""
    agent_id = k.agent_id
    job_id = args.job_id
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()

        # In production, this would:
        # 1. Deploy escrow contract
        # 2. Transfer USDC to escrow
        # 3. Record the escrow address

        # For now, we'll simulate with a placeholder address
        import uuid

        escrow_address = f"0x{''.join(uuid.uuid4().hex[:40])}"

        job = service.fund_job(
            job_id=job_id,
            actor_id=agent_id,
            escrow_address=escrow_address,
            tx_hash=f"0x{'0' * 64}",  # Placeholder
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job_id": job.id,
                        "escrow_address": job.escrow_address,
                        "status": job.status,
                    },
                    indent=2,
                )
            )
        else:
            print("✅ Job funded successfully!")
            print("")
            print(f"  Job ID:  {job.id}")
            print(f"  Escrow:  {job.escrow_address}")
            print(f"  Status:  {job.status}")
            print("")
            print("Workers can now apply to this job.")
            print(f"View applications: kernle job applications {job.id}")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error funding job: {e}")


def _job_applications(args: "argparse.Namespace", k: "Kernle") -> None:
    """List applications for a job."""
    job_id = args.job_id
    output_json = getattr(args, "json", False)
    agent_id = k.agent_id

    try:
        service = _get_job_service()

        # Verify job exists and user is the client
        job = service.get_job(job_id)

        # Service layer handles authorization via actor_id
        applications = service.list_applications(job_id=job_id, actor_id=agent_id)

        if output_json:
            result = [app.to_dict() for app in applications]
            print(json.dumps(result, indent=2, default=str))
        else:
            if not applications:
                print(f"No applications yet for job: {job.title}")
                return

            print(f"📨 Applications for: {job.title}")
            print("=" * 60)

            for app in applications:
                status_emoji = {
                    "pending": "⏳",
                    "accepted": "✅",
                    "rejected": "❌",
                    "withdrawn": "↩️",
                }.get(app.status, "⚪")

                print(f"\n  {status_emoji} {app.applicant_id}")
                print(f"     ID: {app.id[:8]}...")
                print(f"     Status: {app.status}")
                print(f"     Message: {app.message[:100]}{'...' if len(app.message) > 100 else ''}")
                if app.proposed_deadline:
                    print(f"     Proposed deadline: {app.proposed_deadline.strftime('%Y-%m-%d')}")

            print("")
            print("To accept an application:")
            print(f"  kernle job accept {job_id} <application_id>")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_accept(args: "argparse.Namespace", k: "Kernle") -> None:
    """Accept an application."""
    agent_id = k.agent_id
    job_id = args.job_id
    application_id = args.application_id
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job, application = service.accept_application(
            job_id=job_id,
            application_id=application_id,
            actor_id=agent_id,
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job": job.to_dict(),
                        "application": application.to_dict(),
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print("✅ Application accepted!")
            print("")
            print(f"  Job:      {job.title}")
            print(f"  Worker:   {job.worker_id}")
            print(f"  Status:   {job.status}")
            print("")
            print("The worker can now start on the job.")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_approve(args: "argparse.Namespace", k: "Kernle") -> None:
    """Approve deliverable and release payment."""
    agent_id = k.agent_id
    job_id = args.job_id
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job = service.approve_job(
            job_id=job_id,
            actor_id=agent_id,
            tx_hash=f"0x{'0' * 64}",  # Placeholder
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job": job.to_dict(),
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print("✅ Job approved! Payment released.")
            print("")
            print(f"  Job:    {job.title}")
            print(f"  Worker: {job.worker_id}")
            print(f"  Amount: ${job.budget_usdc:.2f} USDC")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_cancel(args: "argparse.Namespace", k: "Kernle") -> None:
    """Cancel a job."""
    agent_id = k.agent_id
    job_id = args.job_id
    reason = getattr(args, "reason", None)
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job = service.cancel_job(
            job_id=job_id,
            actor_id=agent_id,
            reason=reason,
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job_id": job.id,
                        "status": job.status,
                    },
                    indent=2,
                )
            )
        else:
            print("✅ Job cancelled.")
            print("")
            print(f"  Job:    {job.title}")
            print(f"  Status: {job.status}")
            if job.escrow_address:
                print("")
                print("Note: If the job was funded, escrow funds will be refunded.")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_dispute(args: "argparse.Namespace", k: "Kernle") -> None:
    """Raise a dispute on a job."""
    agent_id = k.agent_id
    job_id = args.job_id
    reason = args.reason
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job = service.dispute_job(
            job_id=job_id,
            actor_id=agent_id,
            reason=reason,
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job": job.to_dict(),
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print("⚠️  Dispute raised.")
            print("")
            print(f"  Job:    {job.title}")
            print(f"  Status: {job.status}")
            print(f"  Reason: {reason}")
            print("")
            print("An arbitrator will review and resolve this dispute.")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


# =============================================================================
# Worker Commands
# =============================================================================


def _job_search(args: "argparse.Namespace", k: "Kernle") -> None:
    """Search for available jobs."""
    query = getattr(args, "query", None)
    skills = getattr(args, "skill", None) or []
    min_budget = getattr(args, "min_budget", None)
    max_budget = getattr(args, "max_budget", None)
    limit = getattr(args, "limit", 20)
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        jobs = service.search_jobs(
            query=query,
            skills=skills,
            min_budget=min_budget,
            max_budget=max_budget,
            limit=limit,
        )

        if output_json:
            result = [job.to_dict() for job in jobs]
            print(json.dumps(result, indent=2, default=str))
        else:
            if not jobs:
                print("No jobs found matching your criteria.")
                return

            print(f"🔍 Available Jobs ({len(jobs)} found)")
            print("=" * 60)

            for job in jobs:
                deadline_str = job.deadline.strftime("%Y-%m-%d") if job.deadline else "N/A"
                print(f"\n  💼 {job.title}")
                print(
                    f"     ID: {job.id[:8]}... | ${job.budget_usdc:.0f} USDC | Due: {deadline_str}"
                )
                if job.skills_required:
                    print(f"     Skills: {', '.join(job.skills_required)}")
                print(f"     {job.description[:100]}{'...' if len(job.description) > 100 else ''}")

            print("")
            print("To apply for a job:")
            print("  kernle job apply <job_id> --message 'Your application message'")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")


def _job_apply(args: "argparse.Namespace", k: "Kernle") -> None:
    """Apply to a job."""
    agent_id = k.agent_id
    job_id = args.job_id
    message = args.message
    proposed_deadline = getattr(args, "deadline", None)
    output_json = getattr(args, "json", False)

    try:
        deadline_dt = None
        if proposed_deadline:
            deadline_dt = _parse_deadline(proposed_deadline)

        service = _get_job_service()
        application = service.apply_to_job(
            job_id=job_id,
            applicant_id=agent_id,
            message=message,
            proposed_deadline=deadline_dt,
        )

        if output_json:
            print(json.dumps(application.to_dict(), indent=2, default=str))
        else:
            print("✅ Application submitted!")
            print("")
            print(f"  Application ID: {application.id}")
            print(f"  Job ID:         {job_id}")
            print(f"  Status:         {application.status}")
            print("")
            print("You'll be notified when the client reviews your application.")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            if "already applied" in str(e).lower():
                print("⚠️  You've already applied to this job.")
            elif "not accepting" in str(e).lower():
                print("⚠️  This job is not accepting applications.")
            else:
                print(f"❌ Error: {e}")


def _job_deliver(args: "argparse.Namespace", k: "Kernle") -> None:
    """Submit a deliverable for a job."""
    agent_id = k.agent_id
    job_id = args.job_id
    url = args.url
    hash_val = getattr(args, "hash", None)
    output_json = getattr(args, "json", False)

    try:
        service = _get_job_service()
        job = service.deliver_job(
            job_id=job_id,
            actor_id=agent_id,
            deliverable_url=url,
            deliverable_hash=hash_val,
        )

        if output_json:
            print(
                json.dumps(
                    {
                        "success": True,
                        "job": job.to_dict(),
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            print("✅ Deliverable submitted!")
            print("")
            print(f"  Job:    {job.title}")
            print(f"  URL:    {url}")
            if hash_val:
                print(f"  Hash:   {hash_val}")
            print(f"  Status: {job.status}")
            print("")
            print("The client will review your deliverable.")
            print("Payment will be released upon approval.")

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")
