import pytest
import pandas as pd
from src.pii_engine import PIIDetector

@pytest.fixture
def detector():
    """Fixture to initialize the detector once for all tests."""
    return PIIDetector()

def test_email_detection(detector):
    """Test if the Regex pattern correctly identifies emails."""
    sample_series = pd.Series(["Contact us at info@eib.org", "No email here", "test.user@gmail.com"])
    results = detector.scan_column(sample_series)
    assert "EMAIL" in results

def test_iban_detection(detector):
    """Test if the Regex pattern identifies European IBANs."""
    sample_series = pd.Series(["My IBAN is LU071234567890123456", "Ref: 12345"])
    results = detector.scan_column(sample_series)
    assert "IBAN" in results

def test_person_detection(detector):
    """Test if spaCy's NER identifies human names."""
    sample_series = pd.Series(["Project managed by Lahari Prabha", "Project managed by Purnima Prabha"])
    results = detector.scan_column(sample_series)
    assert "PERSON" in results

def test_clean_column(detector):
    """Test that non-sensitive data returns an empty list."""
    sample_series = pd.Series(["Retail", "Latin America", "2025"])
    results = detector.scan_column(sample_series)
    assert len(results) == 0

def test_mixed_pii_detection(detector):
    """Test a column containing multiple types of PII."""
    sample_series = pd.Series(["John Doe", "john.doe@eib.org", "+35212345678"])
    results = detector.scan_column(sample_series)
    # Check that it caught at least two different types
    assert len(results) >= 2
    assert "PERSON" in results or "EMAIL" in results