import pytest

from pyrite.privacy import PrivacyPolicy


def test_offline_policy_blocks_network():
    policy = PrivacyPolicy()
    with pytest.raises(PermissionError):
        policy.assert_network_allowed()


def test_offline_policy_rejects_telemetry():
    with pytest.raises(ValueError):
        PrivacyPolicy(offline_only=True, telemetry=True).validate()
