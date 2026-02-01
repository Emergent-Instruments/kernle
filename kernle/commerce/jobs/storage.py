"""
Jobs storage layer.

Provides persistence for jobs and job applications using Supabase backend.
"""

from datetime import datetime, timezone
from typing import List, Optional, Protocol
import logging

from kernle.commerce.jobs.models import (
    Job,
    JobApplication,
    JobStatus,
    ApplicationStatus,
    JobStateTransition,
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
        
        return jobs[offset:offset + limit]
    
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


class InMemoryJobStorage:
    """In-memory job storage for testing and development.
    
    Note: Data is not persisted between process restarts.
    This is useful for testing before the Supabase backend is integrated.
    """
    
    def __init__(self):
        from typing import Dict
        self._jobs: Dict[str, Job] = {}
        self._applications: Dict[str, JobApplication] = {}
        self._transitions: Dict[str, List[JobStateTransition]] = {}
    
    def _utc_now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    # === Jobs ===
    
    def save_job(self, job: Job) -> str:
        """Save a job listing."""
        self._jobs[job.id] = job
        logger.debug(f"Saved job {job.id}: {job.title}")
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
        if status:
            status_val = status.value if isinstance(status, JobStatus) else status
            jobs = [j for j in jobs if j.status == status_val]
        
        if client_id:
            jobs = [j for j in jobs if j.client_id == client_id]
        
        if worker_id:
            jobs = [j for j in jobs if j.worker_id == worker_id]
        
        if skills:
            # Match if job requires any of the specified skills
            jobs = [j for j in jobs if any(s in j.skills_required for s in skills)]
        
        if min_budget is not None:
            jobs = [j for j in jobs if j.budget_usdc >= min_budget]
        
        if max_budget is not None:
            jobs = [j for j in jobs if j.budget_usdc <= max_budget]
        
        # Sort by created_at (newest first)
        jobs.sort(key=lambda j: j.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        # Apply pagination
        return jobs[offset:offset + limit]
    
    def update_job(self, job: Job) -> bool:
        """Update a job."""
        if job.id not in self._jobs:
            return False
        self._jobs[job.id] = job
        logger.debug(f"Updated job {job.id}")
        return True
    
    # === Applications ===
    
    def save_application(self, application: JobApplication) -> str:
        """Save a job application."""
        self._applications[application.id] = application
        logger.debug(f"Saved application {application.id} for job {application.job_id}")
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
        applications = list(self._applications.values())
        
        if job_id:
            applications = [a for a in applications if a.job_id == job_id]
        
        if applicant_id:
            applications = [a for a in applications if a.applicant_id == applicant_id]
        
        if status:
            status_val = status.value if isinstance(status, ApplicationStatus) else status
            applications = [a for a in applications if a.status == status_val]
        
        # Sort by created_at (newest first)
        applications.sort(
            key=lambda a: a.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )
        
        return applications[:limit]
    
    def update_application_status(
        self,
        application_id: str,
        status: ApplicationStatus,
    ) -> bool:
        """Update application status."""
        application = self._applications.get(application_id)
        if not application:
            return False
        
        status_val = status.value if isinstance(status, ApplicationStatus) else status
        application.status = status_val
        logger.debug(f"Updated application {application_id} status to {status_val}")
        return True
    
    # === Transitions ===
    
    def save_transition(self, transition: JobStateTransition) -> str:
        """Save a state transition record."""
        if transition.job_id not in self._transitions:
            self._transitions[transition.job_id] = []
        self._transitions[transition.job_id].append(transition)
        logger.debug(
            f"Saved transition for job {transition.job_id}: "
            f"{transition.from_status} → {transition.to_status}"
        )
        return transition.id
    
    def get_transitions(self, job_id: str) -> List[JobStateTransition]:
        """Get all state transitions for a job."""
        transitions = self._transitions.get(job_id, [])
        # Sort by created_at (oldest first)
        transitions.sort(
            key=lambda t: t.created_at or datetime.min.replace(tzinfo=timezone.utc)
        )
        return transitions


class SupabaseJobStorage:
    """Supabase-backed job storage.
    
    Note: This is a placeholder implementation. The actual Supabase
    integration will be added when the backend routes are implemented.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize Supabase connection."""
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self._client = None
    
    def _utc_now(self) -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    # === Jobs ===
    
    def save_job(self, job: Job) -> str:
        """Save a job listing."""
        logger.info(f"Saving job {job.id}: {job.title}")
        # TODO: Implement Supabase insert
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        logger.debug(f"Getting job {job_id}")
        # TODO: Implement Supabase query
        return None
    
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
        logger.debug(f"Listing jobs with filters: status={status}, client={client_id}")
        # TODO: Implement Supabase query with filters
        # query = self._client.table("jobs").select("*")
        # if status:
        #     query = query.eq("status", status.value)
        # if client_id:
        #     query = query.eq("client_id", client_id)
        # if worker_id:
        #     query = query.eq("worker_id", worker_id)
        # if skills:
        #     query = query.overlaps("skills_required", skills)
        # if min_budget:
        #     query = query.gte("budget_usdc", min_budget)
        # if max_budget:
        #     query = query.lte("budget_usdc", max_budget)
        # query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        return []
    
    def update_job(self, job: Job) -> bool:
        """Update a job."""
        logger.info(f"Updating job {job.id}")
        # TODO: Implement Supabase update
        return True
    
    # === Applications ===
    
    def save_application(self, application: JobApplication) -> str:
        """Save a job application."""
        logger.info(f"Saving application {application.id} for job {application.job_id}")
        # TODO: Implement Supabase insert
        return application.id
    
    def get_application(self, application_id: str) -> Optional[JobApplication]:
        """Get an application by ID."""
        logger.debug(f"Getting application {application_id}")
        # TODO: Implement Supabase query
        return None
    
    def list_applications(
        self,
        job_id: Optional[str] = None,
        applicant_id: Optional[str] = None,
        status: Optional[ApplicationStatus] = None,
        limit: int = 100,
    ) -> List[JobApplication]:
        """List applications with optional filters."""
        logger.debug(f"Listing applications for job={job_id}, applicant={applicant_id}")
        # TODO: Implement Supabase query
        return []
    
    def update_application_status(
        self,
        application_id: str,
        status: ApplicationStatus,
    ) -> bool:
        """Update application status."""
        logger.info(f"Updating application {application_id} status to {status.value}")
        # TODO: Implement Supabase update
        return True
    
    # === Transitions ===
    
    def save_transition(self, transition: JobStateTransition) -> str:
        """Save a state transition record."""
        logger.info(
            f"Saving transition for job {transition.job_id}: "
            f"{transition.from_status} → {transition.to_status}"
        )
        # TODO: Implement Supabase insert
        return transition.id
    
    def get_transitions(self, job_id: str) -> List[JobStateTransition]:
        """Get all state transitions for a job."""
        logger.debug(f"Getting transitions for job {job_id}")
        # TODO: Implement Supabase query
        return []
