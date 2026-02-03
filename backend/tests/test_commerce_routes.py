"""Tests for commerce API routes."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest


class TestWalletRoutes:
    """Tests for wallet endpoints."""

    @pytest.mark.skip(
        reason="Requires integration test - auth middleware connects to Supabase before patches"
    )
    def test_get_my_wallet_success(self, client, auth_headers):
        """Test getting own wallet."""
        mock_wallet = {
            "id": "wallet-123",
            "agent_id": "test-agent",
            "wallet_address": "0x1234567890123456789012345678901234567890",
            "chain": "base",
            "status": "active",
            "owner_eoa": "0x0987654321098765432109876543210987654321",
            "spending_limit_per_tx": 100.0,
            "spending_limit_daily": 1000.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("app.database.get_user", new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = {
                "user_id": "usr_TEST_ONLY_000000",
                "tier": "free",
                "is_admin": False,
            }
            with patch(
                "app.routes.commerce.wallets.get_wallet_by_agent", new_callable=AsyncMock
            ) as mock_by_agent:
                mock_by_agent.return_value = mock_wallet
                response = client.get("/api/v1/wallets/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["wallet_address"] == "0x1234567890123456789012345678901234567890"
        assert data["status"] == "active"

    def test_get_my_wallet_not_found(self, client, auth_headers):
        """Test wallet not found error."""
        with patch(
            "app.routes.commerce.wallets.get_wallet_by_agent", new_callable=AsyncMock
        ) as mock_by_agent:
            with patch(
                "app.routes.commerce.wallets.get_wallet_by_user", new_callable=AsyncMock
            ) as mock_by_user:
                mock_by_agent.return_value = None
                mock_by_user.return_value = None
                response = client.get("/api/v1/wallets/me", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestJobRoutes:
    """Tests for job endpoints."""

    def test_create_job_success(self, client, auth_headers):
        """Test creating a job listing."""
        future_deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        mock_job = {
            "id": "job-123",
            "client_id": "test-agent",
            "title": "Test Job",
            "description": "A test job description",
            "budget_usdc": 100.0,
            "deadline": future_deadline,
            "skills_required": ["coding"],
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("app.routes.commerce.jobs.create_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_job
            response = client.post(
                "/api/v1/jobs",
                json={
                    "title": "Test Job",
                    "description": "A test job description",
                    "budget_usdc": 100.0,
                    "deadline": future_deadline,
                    "skills_required": ["coding"],
                },
                headers=auth_headers,
            )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Job"
        assert data["status"] == "open"

    def test_create_job_deadline_in_past(self, client, auth_headers):
        """Test creating a job with past deadline fails."""
        past_deadline = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        response = client.post(
            "/api/v1/jobs",
            json={
                "title": "Test Job",
                "description": "A test job description",
                "budget_usdc": 100.0,
                "deadline": past_deadline,
            },
            headers=auth_headers,
        )

        assert response.status_code == 422  # Validation error

    def test_list_jobs(self, client, auth_headers):
        """Test listing jobs."""
        mock_jobs = [
            {
                "id": "job-1",
                "client_id": "agent-1",
                "title": "Job 1",
                "description": "Desc 1",
                "budget_usdc": 50.0,
                "deadline": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                "skills_required": [],
                "status": "open",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        with patch("app.routes.commerce.jobs.list_jobs", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = (mock_jobs, 1)
            response = client.get("/api/v1/jobs", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["jobs"]) == 1

    def test_get_job_details(self, client, auth_headers):
        """Test getting job details."""
        mock_job = {
            "id": "job-123",
            "client_id": "agent-1",
            "title": "Test Job",
            "description": "Description",
            "budget_usdc": 100.0,
            "deadline": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
            "skills_required": ["coding"],
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("app.routes.commerce.jobs.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_job
            response = client.get("/api/v1/jobs/job-123", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "job-123"

    def test_get_job_not_found(self, client, auth_headers):
        """Test job not found error."""
        with patch("app.routes.commerce.jobs.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = client.get("/api/v1/jobs/nonexistent", headers=auth_headers)

        assert response.status_code == 404

    def test_apply_to_job_success(self, client, auth_headers):
        """Test applying to a job."""
        mock_job = {
            "id": "job-123",
            "client_id": "different-agent",  # Not the authenticated user
            "status": "funded",
        }
        mock_app = {
            "id": "app-123",
            "job_id": "job-123",
            "applicant_id": "test-agent",
            "message": "I'm interested!",
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("app.routes.commerce.jobs.get_job", new_callable=AsyncMock) as mock_get:
            with patch(
                "app.routes.commerce.jobs.create_application", new_callable=AsyncMock
            ) as mock_create:
                mock_get.return_value = mock_job
                mock_create.return_value = mock_app
                response = client.post(
                    "/api/v1/jobs/job-123/apply",
                    json={"message": "I'm interested!"},
                    headers=auth_headers,
                )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"

    def test_apply_to_own_job_fails(self, client, auth_headers):
        """Test that applying to own job fails."""
        # The auth_headers fixture uses usr_TEST_ONLY_000000
        mock_job = {
            "id": "job-123",
            "client_id": "usr_TEST_ONLY_000000",  # Same as authenticated user
            "status": "funded",
        }

        with patch("app.routes.commerce.jobs.get_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_job
            response = client.post(
                "/api/v1/jobs/job-123/apply",
                json={"message": "I'm interested!"},
                headers=auth_headers,
            )

        assert response.status_code == 400
        assert "own job" in response.json()["detail"].lower()


class TestSkillRoutes:
    """Tests for skill endpoints."""

    def test_list_skills(self, client, auth_headers):
        """Test listing skills."""
        mock_skills = [
            {
                "id": "skill-1",
                "name": "coding",
                "description": "Software development",
                "category": "technical",
                "usage_count": 10,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        with patch("app.routes.commerce.skills.list_skills", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = (mock_skills, 1)
            response = client.get("/api/v1/skills", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["skills"][0]["name"] == "coding"

    def test_get_skill_details(self, client, auth_headers):
        """Test getting skill details."""
        mock_skill = {
            "id": "skill-1",
            "name": "coding",
            "description": "Software development",
            "category": "technical",
            "usage_count": 10,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch("app.routes.commerce.skills.get_skill", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_skill
            response = client.get("/api/v1/skills/coding", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "coding"

    def test_find_agents_with_skill(self, client, auth_headers):
        """Test finding agents with a skill."""
        mock_skill = {"id": "skill-1", "name": "coding"}

        with patch("app.routes.commerce.skills.get_skill", new_callable=AsyncMock) as mock_get:
            with patch(
                "app.routes.commerce.skills.find_agents_with_skill", new_callable=AsyncMock
            ) as mock_find:
                mock_get.return_value = mock_skill
                mock_find.return_value = ["agent-1", "agent-2"]
                response = client.get("/api/v1/skills/coding/agents", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["skill"] == "coding"
        assert data["total"] == 2


class TestEscrowRoutes:
    """Tests for escrow endpoints."""

    def test_get_escrow_details(self, client, auth_headers):
        """Test getting escrow details."""
        mock_job = {
            "id": "job-123",
            "client_id": "agent-1",
            "worker_id": "agent-2",
            "budget_usdc": 100.0,
            "status": "completed",
            "funded_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        with patch(
            "app.routes.commerce.escrow.get_job_by_escrow", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_job
            response = client.get(
                "/api/v1/escrow/0x1234567890123456789012345678901234567890",
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "released"  # completed -> released
        assert data["job_id"] == "job-123"

    def test_get_escrow_not_found(self, client, auth_headers):
        """Test escrow not found."""
        with patch(
            "app.routes.commerce.escrow.get_job_by_escrow", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None
            response = client.get(
                "/api/v1/escrow/0x1234567890123456789012345678901234567890",
                headers=auth_headers,
            )

        assert response.status_code == 404

    def test_get_escrow_invalid_address(self, client, auth_headers):
        """Test invalid escrow address format."""
        response = client.get(
            "/api/v1/escrow/invalid-address",
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()


class TestMaintenanceRoutes:
    """Tests for maintenance (timeout enforcement) endpoints."""

    def test_maintenance_health_no_issues(self, client, auth_headers):
        """Test maintenance health when no jobs need action."""
        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            with patch(
                "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
            ) as mock_disputes:
                mock_deadline.return_value = []
                mock_disputes.return_value = []
                response = client.get("/api/v1/maintenance/health", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["jobs_past_deadline"] == 0
        assert data["disputes_past_timeout"] == 0

    def test_maintenance_health_action_needed(self, client, auth_headers):
        """Test maintenance health when jobs need action."""
        mock_overdue = [{"id": "job-1", "deadline": "2024-01-01T00:00:00Z"}]
        mock_disputes = [{"id": "job-2", "disputed_at": "2024-01-01T00:00:00Z"}]

        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            with patch(
                "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
            ) as mock_disp:
                mock_deadline.return_value = mock_overdue
                mock_disp.return_value = mock_disputes
                response = client.get("/api/v1/maintenance/health", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "action_needed"
        assert data["jobs_past_deadline"] == 1
        assert data["disputes_past_timeout"] == 1

    def test_check_timeouts_dry_run(self, client, auth_headers):
        """Test timeout check in dry run mode."""
        mock_overdue = [
            {
                "id": "job-1",
                "deadline": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
                "status": "accepted",
            }
        ]

        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            with patch(
                "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
            ) as mock_disputes:
                mock_deadline.return_value = mock_overdue
                mock_disputes.return_value = []
                response = client.post(
                    "/api/v1/maintenance/check-timeouts",
                    json={"dry_run": True},
                    headers=auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is True
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action"] == "would_cancel"
        assert data["actions"][0]["previous_status"] == "accepted"

    def test_check_timeouts_executes_cancellation(self, client, auth_headers):
        """Test timeout check actually cancels jobs when not dry run."""
        mock_overdue = [
            {
                "id": "job-1",
                "deadline": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
                "status": "accepted",
            }
        ]

        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            with patch(
                "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
            ) as mock_disputes:
                with patch(
                    "app.routes.commerce.maintenance.cancel_job_for_timeout", new_callable=AsyncMock
                ) as mock_cancel:
                    mock_deadline.return_value = mock_overdue
                    mock_disputes.return_value = []
                    mock_cancel.return_value = {"id": "job-1", "status": "cancelled"}

                    response = client.post(
                        "/api/v1/maintenance/check-timeouts",
                        json={"dry_run": False},
                        headers=auth_headers,
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is False
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action"] == "cancelled"
        mock_cancel.assert_called_once()

    def test_check_timeouts_dispute_escalation(self, client, auth_headers):
        """Test dispute timeout escalation."""
        mock_stale_dispute = [
            {
                "id": "job-2",
                "disputed_at": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
                "status": "disputed",
            }
        ]

        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            with patch(
                "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
            ) as mock_disputes:
                with patch(
                    "app.routes.commerce.maintenance.cancel_job_for_timeout", new_callable=AsyncMock
                ) as mock_cancel:
                    mock_deadline.return_value = []
                    mock_disputes.return_value = mock_stale_dispute
                    mock_cancel.return_value = {"id": "job-2", "status": "cancelled"}

                    response = client.post(
                        "/api/v1/maintenance/check-timeouts",
                        json={"dry_run": False, "dispute_timeout_days": 14},
                        headers=auth_headers,
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["total_dispute_timeouts"] == 1
        assert len(data["actions"]) == 1
        assert data["actions"][0]["previous_status"] == "disputed"

    def test_list_overdue_jobs(self, client, auth_headers):
        """Test listing overdue jobs."""
        mock_overdue = [
            {
                "id": "job-1",
                "title": "Test Job",
                "client_id": "client-1",
                "worker_id": "worker-1",
                "deadline": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
                "accepted_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
                "budget_usdc": 100.0,
            }
        ]

        with patch(
            "app.routes.commerce.maintenance.get_jobs_past_deadline", new_callable=AsyncMock
        ) as mock_deadline:
            mock_deadline.return_value = mock_overdue
            response = client.get("/api/v1/maintenance/overdue-jobs", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "job-1"

    def test_list_stale_disputes(self, client, auth_headers):
        """Test listing stale disputes."""
        mock_disputes = [
            {
                "id": "job-2",
                "title": "Disputed Job",
                "client_id": "client-1",
                "worker_id": "worker-1",
                "disputed_at": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
                "budget_usdc": 200.0,
            }
        ]

        with patch(
            "app.routes.commerce.maintenance.get_disputes_past_timeout", new_callable=AsyncMock
        ) as mock_disp:
            mock_disp.return_value = mock_disputes
            response = client.get("/api/v1/maintenance/stale-disputes", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "job-2"


class TestJobStateTransitions:
    """Tests for job state machine transitions."""

    def test_disputed_to_cancelled_transition_allowed(self, client, auth_headers):
        """Test that disputed -> cancelled transition is valid."""
        from app.routes.commerce.jobs import can_transition

        assert can_transition("disputed", "cancelled") is True

    def test_valid_transitions(self, client, auth_headers):
        """Test valid state transitions."""
        from app.routes.commerce.jobs import can_transition

        # Valid transitions
        assert can_transition("open", "funded") is True
        assert can_transition("open", "cancelled") is True
        assert can_transition("funded", "accepted") is True
        assert can_transition("funded", "cancelled") is True
        assert can_transition("accepted", "delivered") is True
        assert can_transition("accepted", "disputed") is True
        assert can_transition("accepted", "cancelled") is True
        assert can_transition("delivered", "completed") is True
        assert can_transition("delivered", "disputed") is True
        assert can_transition("disputed", "completed") is True
        assert can_transition("disputed", "cancelled") is True  # NEW: timeout/escalation path

    def test_invalid_transitions(self, client, auth_headers):
        """Test invalid state transitions."""
        from app.routes.commerce.jobs import can_transition

        # Invalid transitions
        assert can_transition("open", "completed") is False
        assert can_transition("funded", "completed") is False
        assert can_transition("completed", "disputed") is False
        assert can_transition("cancelled", "open") is False
        assert can_transition("delivered", "cancelled") is False
