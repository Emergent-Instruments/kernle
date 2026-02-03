"""Jobs marketplace subsystem for Kernle Commerce.

Provides the jobs marketplace for agents to post and complete work.

Models:
- Job: A work listing in the marketplace
- JobApplication: An application to work on a job
- JobStatus: Job lifecycle status
- ApplicationStatus: Application lifecycle status
- JobStateTransition: Audit log entry for state changes

Service:
- JobService: Job operations (create, apply, accept, deliver, etc.)
"""

from kernle.commerce.jobs.models import (
    VALID_JOB_TRANSITIONS,
    ApplicationStatus,
    Job,
    JobApplication,
    JobStateTransition,
    JobStatus,
)
from kernle.commerce.jobs.service import (
    ApplicationNotFoundError,
    DuplicateApplicationError,
    InvalidTransitionError,
    JobExpiredError,
    JobNotFoundError,
    JobSearchFilters,
    JobService,
    JobServiceError,
    UnauthorizedError,
)

__all__ = [
    # Models
    "Job",
    "JobApplication",
    "JobStatus",
    "ApplicationStatus",
    "JobStateTransition",
    "VALID_JOB_TRANSITIONS",
    # Service
    "JobService",
    "JobServiceError",
    "JobNotFoundError",
    "ApplicationNotFoundError",
    "InvalidTransitionError",
    "UnauthorizedError",
    "DuplicateApplicationError",
    "JobExpiredError",
    "JobSearchFilters",
]
