"""API keys for library systems and the HMAC signature on result callbacks."""

import hashlib
import hmac

import pytest

from utils.integration_security import (
    SIGNATURE_TOLERANCE_S,
    generate_api_key,
    hash_api_key,
    sign_callback,
    verify_callback,
)


class TestApiKey:
    def test_key_is_prefixed_and_stored_only_as_hash(self):
        key, prefix, digest = generate_api_key()

        assert key.startswith("dfk_") and len(key) >= 40
        assert key.startswith(prefix) and len(prefix) == 12
        assert digest == hash_api_key(key) == hashlib.sha256(key.encode()).hexdigest()
        assert key not in digest

    def test_keys_are_unique(self):
        assert generate_api_key()[0] != generate_api_key()[0]


class TestCallbackSignature:
    SECRET = "whsec_test"
    BODY = b'{"job_id":"JOB_001","status":"COMPLETED"}'

    def test_signature_covers_timestamp_and_body(self):
        sig = sign_callback(self.SECRET, 1_700_000_000, self.BODY)
        expected = hmac.new(
            self.SECRET.encode(), b"1700000000." + self.BODY, hashlib.sha256
        ).hexdigest()
        assert sig == f"sha256={expected}"

    def test_receiver_accepts_a_fresh_genuine_callback(self):
        sig = sign_callback(self.SECRET, 1_700_000_000, self.BODY)
        assert verify_callback(self.SECRET, "1700000000", self.BODY, sig, now=1_700_000_010)

    @pytest.mark.parametrize(
        "secret,body,ts,now",
        [
            ("other", BODY, "1700000000", 1_700_000_000),  # wrong secret
            (SECRET, BODY + b" ", "1700000000", 1_700_000_000),  # body tampered
            (SECRET, BODY, "1700000001", 1_700_000_001),  # timestamp swapped
            (SECRET, BODY, "1700000000", 1_700_000_000 + SIGNATURE_TOLERANCE_S + 1),  # replay
            (SECRET, BODY, "not-a-number", 1_700_000_000),
        ],
    )
    def test_receiver_rejects(self, secret, body, ts, now):
        sig = sign_callback(self.SECRET, 1_700_000_000, self.BODY)
        assert not verify_callback(secret, ts, body, sig, now=now)
