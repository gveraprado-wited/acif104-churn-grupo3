import pytest

from backend import customer_lookup
from backend.inference import ChurnPredictor


@pytest.fixture(scope="session")
def predictor():
    """A single real ChurnPredictor (no mocks) for the whole test session:
    loading the real artifacts is fast, and they're the same ones the app uses.
    """
    return ChurnPredictor()


@pytest.fixture(scope="session")
def valid_customer() -> dict:
    """The 30 raw variables of a real test-set customer."""
    customer_id = customer_lookup.list_test_customer_ids()[0]
    return customer_lookup.get_test_customer(customer_id)
