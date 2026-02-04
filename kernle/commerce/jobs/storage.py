"""
Jobs storage layer.

Provides persistence for jobs and job applications using Supabase backend.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Protocol

from kernle.commerce.jobs.models import (
    ApplicationStatus,
    Job,
    JobApplication,
    JobStateTransition,
    JobStatus,
)

logger = logging.getLogger(__name__)


class JobStorage(Protocol):
    """Protocol for job persistence backends."""

    # Jobs
    def save_job(self, job: Job) -> str:
        """Save a job listing. Returns the job ID."""
        ...

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        ...

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        client_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filters."""
        ...

    def update_job(self, job: Job) -> bool:
        """Update a job. Returns True if successful."""
        ...

    # Applications
    def save_application(self, application: JobApplication) -> str:
        """Save a job application. Returns the application ID."""
        ...

    def get_application(self, application_id: str) -> Optional[JobApplication]:
        """Get an application by ID."""
        ...

    def list_applications(
        self,
        job_id: Optional[str] = None,
        applicant_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
    ) -> List[JobApplication]:
        """List applications with optional filters."""
        ...

    def update_application_status(
        self,
        application_id: str,
        status: ApplicationStatus,
    ) -> bool:
        """Update application status."""
        ...

    # Transitions (audit log)
    def save_transition(self, transition: JobStateTransition) -> str:
        """Save a state transition record. Returns the transition ID."""
        ...

    def get_transitions(self, job_id: str) -> List[JobStateTransition]:
        """Get all state transitions for a job."""
        ...


class InMemoryJobStorage:
    """In-memory job storage for testing and local development."""

    def __init__(self):
        """Initialize empty storage."""
        self._jobs: dict[str, Job] = {}
        self._applications: dict[str, JobApplication] = {}
        self._transitions: dict[str, list[JobStateTransition]] = {}  # job_id -> list

    def _utc_now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)

    # === Jobs ===

    def save_job(self, job: Job) -> str:
        """Save a job listing."""
        self._jobs[job.id] = job
        if job.id not in self._transitions:
            self._transitions[job.id] = []
        return job.id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        client_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        skills: Optional[List[str]] = None,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filters."""
        jobs = list(self._jobs.values())

        # Apply filters
        if status is not None:
            status_val = status.value if isinstance(status, JobStatus) else status
            jobs = [j for j in jobs if j.status == status_val]
        if client_id is not None:
            jobs = [j for j in jobs if j.client_id == client_id]
        if worker_id is not None:
            jobs = [j for j in jobs if j.worker_id == worker_id]
        if skills is not None:
            jobs = [j for j in jobs if any(s in j.skills_required for s in skills)]
        if min_budget is not None:
            jobs = [j for j in jobs if j.budget_usdc >= min_budget]
        if max_budget is not None:
            jobs = [j for j in jobs if j.budget_usdc <= max_budget]

        # Sort by created_at desc
        jobs.sort(key=lambda j: j.created_at or self._utc_now(), reverse=True)

        return jobs[offset : offset + limit]

    def update_job(self, job: Job) -> bool:
        """Update a job."""
        if job.id not in self._jobs:
            return False
        self._jobs[job.id] = job
        return True

    # === Applications ===

    def save_application(self, application: JobApplication) -> str:
        """Save a job application."""
        self._applications[application.id] = application
        return application.id

    def get_application(self, application_id: str) -> Optional[JobApplication]:
        """Get an application by ID."""
        return self._applications.get(application_id)

    def list_applications(
        self,
        job_id: Optional[str] = None,
        applicant_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
    ) -> List[JobApplication]:
        """List applications with optional filters."""
        apps = list(self._applications.values())

        if job_id is not None:
            apps = [a for a in apps if a.job_id == job_id]
        if applicant_id is not None:
            apps = [a for a in apps if a.applicant_id == applicant_id]
        if status is not None:
            status_val = status.value if isinstance(status, ApplicationStatus) else status
            apps = [a for a in apps if a.status == status_val]

        # Sort by created_at desc
        apps.sort(key=lambda a: a.created_at or self._utc_now(), reverse=True)

        return apps[:limit]

    def update_application_status(
        self,
        application_id: str,
        status: ApplicationStatus,
    ) -> bool:
        """Update application status."""
        app = self._applications.get(application_id)
        if not app:
            return False
        app.status = status.value
        return True

    # === Transitions ===

    def save_transition(self, transition: JobStateTransition) -> str:
        """Save a state transition record."""
        if transition.job_id not in self._transitions:
            self._transitions[transition.job_id] = []
        self._transitions[transition.job_id].append(transition)
        return transition.id

    def get_transitions(self, job_id: str) -> List[JobStateTransition]:
        """Get all state transitions for a job."""
        transitions = self._transitions.get(job_id, [])
        # Sort by created_at asc
        return sorted(transitions, key=lambda t: t.created_at or self._utc_now())
