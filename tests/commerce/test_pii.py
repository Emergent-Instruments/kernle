"""Tests for PII detection and redaction."""

from kernle.commerce.pii import (
    PIIType,
    contains_pii,
    detect_pii,
    redact_pii,
)


class TestPIIDetection:
    """Tests for detect_pii function."""

    def test_detect_email(self):
        """Should detect email addresses."""
        text = "Contact me at john.doe@example.com for details"
        findings = detect_pii(text, types=[PIIType.EMAIL])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.EMAIL
        assert findings[0].value == "john.doe@example.com"

    def test_detect_multiple_emails(self):
        """Should detect multiple email addresses."""
        text = "Email john@example.com or jane@test.org"
        findings = detect_pii(text, types=[PIIType.EMAIL])

        assert len(findings) == 2
        assert findings[0].value == "john@example.com"
        assert findings[1].value == "jane@test.org"

    def test_detect_ssn_with_dashes(self):
        """Should detect SSN with dashes."""
        text = "My SSN is 456-78-9012"
        findings = detect_pii(text, types=[PIIType.SSN])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.SSN
        assert "456" in findings[0].value

    def test_detect_ssn_with_spaces(self):
        """Should detect SSN with spaces."""
        text = "SSN: 456 78 9012"
        findings = detect_pii(text, types=[PIIType.SSN])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.SSN

    def test_skip_fake_ssn(self):
        """Should skip obviously fake SSNs."""
        text = "Test SSN 123-45-6789 and 000-00-0000"
        findings = detect_pii(text, types=[PIIType.SSN])

        # 123-45-6789 should be skipped (test pattern)
        # 000-00-0000 should be skipped (invalid area)
        assert len(findings) == 0

    def test_detect_credit_card_visa(self):
        """Should detect valid Visa card number."""
        # Valid test Visa number (passes Luhn)
        text = "Card: 4111 1111 1111 1111"
        findings = detect_pii(text, types=[PIIType.CREDIT_CARD])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.CREDIT_CARD

    def test_detect_credit_card_mastercard(self):
        """Should detect valid Mastercard number."""
        # Valid test Mastercard (passes Luhn)
        text = "Pay with 5500 0000 0000 0004"
        findings = detect_pii(text, types=[PIIType.CREDIT_CARD])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.CREDIT_CARD

    def test_skip_invalid_credit_card(self):
        """Should skip numbers that fail Luhn check."""
        text = "Not a card: 1234 5678 9012 3456"
        findings = detect_pii(text, types=[PIIType.CREDIT_CARD])

        assert len(findings) == 0

    def test_detect_phone_standard(self):
        """Should detect standard US phone format."""
        text = "Call me at (555) 123-4567"
        findings = detect_pii(text, types=[PIIType.PHONE])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.PHONE

    def test_detect_phone_with_country_code(self):
        """Should detect phone with country code."""
        text = "International: +1-555-123-4567"
        findings = detect_pii(text, types=[PIIType.PHONE])

        assert len(findings) == 1
        assert findings[0].pii_type == PIIType.PHONE

    def test_detect_all_types(self):
        """Should detect multiple PII types in one text."""
        text = """
        Contact: john@example.com
        Phone: (555) 123-4567
        SSN: 456-78-9012
        Card: 4111 1111 1111 1111
        """
        findings = detect_pii(text)

        types_found = {f.pii_type for f in findings}
        assert PIIType.EMAIL in types_found
        assert PIIType.PHONE in types_found
        assert PIIType.SSN in types_found
        assert PIIType.CREDIT_CARD in types_found

    def test_no_false_positives_on_clean_text(self):
        """Should not detect PII in clean text."""
        text = """
        Looking for a Python developer to build a REST API.
        Budget is $500, deadline is next Friday.
        Must have 3+ years experience with Django.
        """
        findings = detect_pii(text)

        assert len(findings) == 0


class TestPIIRedaction:
    """Tests for redact_pii function."""

    def test_redact_email(self):
        """Should redact email addresses."""
        text = "Contact john@example.com"
        result = redact_pii(text, types=[PIIType.EMAIL])

        assert "john@example.com" not in result
        assert "[REDACTED-EMAIL]" in result

    def test_redact_ssn(self):
        """Should redact SSN."""
        text = "SSN: 456-78-9012"
        result = redact_pii(text, types=[PIIType.SSN])

        assert "456-78-9012" not in result
        assert "[REDACTED-SSN]" in result

    def test_redact_credit_card(self):
        """Should redact credit card numbers."""
        text = "Card: 4111 1111 1111 1111"
        result = redact_pii(text, types=[PIIType.CREDIT_CARD])

        assert "4111" not in result
        assert "[REDACTED-CC]" in result

    def test_redact_phone(self):
        """Should redact phone numbers."""
        text = "Call (555) 123-4567"
        result = redact_pii(text, types=[PIIType.PHONE])

        assert "(555) 123-4567" not in result
        assert "[REDACTED-PHONE]" in result

    def test_redact_multiple(self):
        """Should redact multiple PII instances."""
        text = "Email john@test.com or jane@test.com"
        result = redact_pii(text, types=[PIIType.EMAIL])

        assert "john@test.com" not in result
        assert "jane@test.com" not in result
        assert result.count("[REDACTED-EMAIL]") == 2

    def test_preserve_non_pii_text(self):
        """Should preserve text around PII."""
        text = "Contact john@example.com for more info"
        result = redact_pii(text, types=[PIIType.EMAIL])

        assert result == "Contact [REDACTED-EMAIL] for more info"


class TestContainsPII:
    """Tests for contains_pii function."""

    def test_returns_true_for_pii(self):
        """Should return True when PII is present."""
        assert contains_pii("email: test@example.com")
        assert contains_pii("SSN: 456-78-9012")
        assert contains_pii("Card: 4111 1111 1111 1111")
        assert contains_pii("Phone: (555) 123-4567")

    def test_returns_false_for_clean_text(self):
        """Should return False for clean text."""
        assert not contains_pii("Hello, this is a normal message.")
        assert not contains_pii("Budget: $500, deadline: Friday")

    def test_type_filtering(self):
        """Should only check specified types."""
        text = "email: test@example.com"

        assert contains_pii(text, types=[PIIType.EMAIL])
        assert not contains_pii(text, types=[PIIType.SSN])
        assert not contains_pii(text, types=[PIIType.CREDIT_CARD])


class TestJobServicePIIIntegration:
    """Tests for PII handling in JobService."""

    def test_create_job_redacts_pii_by_default(self):
        """Should redact PII from job description by default."""
        from datetime import datetime, timedelta, timezone

        from kernle.commerce.jobs.service import JobService
        from kernle.commerce.jobs.storage import InMemoryJobStorage

        service = JobService(storage=InMemoryJobStorage())
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        job = service.create_job(
            client_id="agent_1",
            title="Test Job",
            description="Contact me at secret@email.com for details",
            budget_usdc=100.0,
            deadline=deadline,
        )

        assert "secret@email.com" not in job.description
        assert "[REDACTED-EMAIL]" in job.description

    def test_create_job_preserves_pii_when_disabled(self):
        """Should preserve PII when redaction is disabled."""
        from datetime import datetime, timedelta, timezone

        from kernle.commerce.jobs.service import JobService
        from kernle.commerce.jobs.storage import InMemoryJobStorage

        service = JobService(storage=InMemoryJobStorage())
        deadline = datetime.now(timezone.utc) + timedelta(days=7)

        job = service.create_job(
            client_id="agent_1",
            title="Test Job",
            description="Contact me at secret@email.com for details",
            budget_usdc=100.0,
            deadline=deadline,
            redact_pii=False,
        )

        assert "secret@email.com" in job.description
        assert "[REDACTED-EMAIL]" not in job.description
