"""PII (Personally Identifiable Information) detection and redaction.

This module provides utilities for detecting and optionally redacting
sensitive information from text content, such as:
- Email addresses
- Social Security Numbers (SSN)
- Credit card numbers
- Phone numbers

Usage:
    from kernle.commerce.pii import detect_pii, redact_pii, PIIType

    # Check if text contains PII
    findings = detect_pii("Contact me at john@example.com")
    if findings:
        print(f"Found PII: {findings}")

    # Redact PII from text
    safe_text = redact_pii("My SSN is 123-45-6789")
    # Returns: "My SSN is [REDACTED-SSN]"
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PIIType(Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PHONE = "phone"


@dataclass
class PIIFinding:
    """A detected PII occurrence."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    redacted: str


# Regex patterns for PII detection
# Note: These are intentionally conservative to reduce false positives

# Email: standard format, requires @ and domain
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# SSN: xxx-xx-xxxx or xxx xx xxxx or xxxxxxxxx (9 digits)
# Excludes obviously fake ones like 000-00-0000, 123-45-6789
SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})([0-8]\d{2})"  # Area number (not 000, 666, 9xx)
    r"[-\s]?"
    r"(?!00)(\d{2})"  # Group number (not 00)
    r"[-\s]?"
    r"(?!0000)(\d{4})\b"  # Serial number (not 0000)
)

# Credit card: 13-19 digits, optionally separated by spaces or dashes
# Validates using Luhn algorithm
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b"
)

# Phone: various US formats
# (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, +1xxxxxxxxxx
PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?"  # Optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"  # Area code
    r"\d{3}[-.\s]?"  # Exchange
    r"\d{4}\b"  # Subscriber
)


def _luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    # Luhn algorithm
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def detect_pii(
    text: str,
    types: Optional[List[PIIType]] = None,
) -> List[PIIFinding]:
    """Detect PII in text.

    Args:
        text: Text to scan for PII
        types: Specific PII types to detect (default: all)

    Returns:
        List of PIIFinding objects for each detected occurrence
    """
    if types is None:
        types = list(PIIType)

    findings: List[PIIFinding] = []

    if PIIType.EMAIL in types:
        for match in EMAIL_PATTERN.finditer(text):
            findings.append(
                PIIFinding(
                    pii_type=PIIType.EMAIL,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    redacted="[REDACTED-EMAIL]",
                )
            )

    if PIIType.SSN in types:
        for match in SSN_PATTERN.finditer(text):
            # Additional validation: not obviously fake
            full_match = match.group()
            digits_only = re.sub(r"[-\s]", "", full_match)
            # Skip test SSNs (123-45-6789, etc.)
            if digits_only not in ("123456789", "111111111", "222222222"):
                findings.append(
                    PIIFinding(
                        pii_type=PIIType.SSN,
                        value=full_match,
                        start=match.start(),
                        end=match.end(),
                        redacted="[REDACTED-SSN]",
                    )
                )

    if PIIType.CREDIT_CARD in types:
        for match in CREDIT_CARD_PATTERN.finditer(text):
            card_number = match.group()
            # Validate with Luhn algorithm to reduce false positives
            if _luhn_check(card_number):
                findings.append(
                    PIIFinding(
                        pii_type=PIIType.CREDIT_CARD,
                        value=card_number,
                        start=match.start(),
                        end=match.end(),
                        redacted="[REDACTED-CC]",
                    )
                )

    if PIIType.PHONE in types:
        for match in PHONE_PATTERN.finditer(text):
            # Skip if it looks like a version number or other non-phone pattern
            phone = match.group()
            digits_only = re.sub(r"[^\d]", "", phone)
            if len(digits_only) >= 10:  # At least 10 digits for a valid phone
                findings.append(
                    PIIFinding(
                        pii_type=PIIType.PHONE,
                        value=phone,
                        start=match.start(),
                        end=match.end(),
                        redacted="[REDACTED-PHONE]",
                    )
                )

    # Sort by position in text
    findings.sort(key=lambda f: f.start)

    return findings


def redact_pii(
    text: str,
    types: Optional[List[PIIType]] = None,
) -> str:
    """Redact PII from text.

    Args:
        text: Text to redact PII from
        types: Specific PII types to redact (default: all)

    Returns:
        Text with PII replaced by redaction markers
    """
    findings = detect_pii(text, types)

    if not findings:
        return text

    # Build result by replacing matches from end to start
    # (to preserve indices)
    result = text
    for finding in reversed(findings):
        result = result[: finding.start] + finding.redacted + result[finding.end :]

    return result


def contains_pii(
    text: str,
    types: Optional[List[PIIType]] = None,
) -> bool:
    """Check if text contains any PII.

    Args:
        text: Text to check
        types: Specific PII types to check (default: all)

    Returns:
        True if any PII is detected
    """
    return len(detect_pii(text, types)) > 0
