"""Tests for PKI/crypto module."""

import tempfile
from pathlib import Path

import pytest

from kernle.comms.crypto import (
    KeyAlreadyExistsError,
    KeyManager,
    KeyNotFoundError,
    KeyPair,
    SignatureError,
    generate_key_pair,
    sign_message,
    verify_signature,
)


class TestKeyPairGeneration:
    """Tests for key pair generation."""

    def test_generate_key_pair_returns_valid_keypair(self):
        """Test that generate_key_pair returns a valid KeyPair."""
        key_pair = generate_key_pair()

        assert key_pair.public_key is not None
        assert key_pair.private_key is not None
        assert key_pair.created_at is not None
        assert key_pair.key_id is not None

    def test_generate_key_pair_keys_are_base64(self):
        """Test that keys are valid base64."""
        import base64

        key_pair = generate_key_pair()

        # Should not raise on decode
        public_bytes = base64.b64decode(key_pair.public_key)
        private_bytes = base64.b64decode(key_pair.private_key)

        # Ed25519 keys are 32 bytes
        assert len(public_bytes) == 32
        assert len(private_bytes) == 32

    def test_generate_key_pair_unique_keys(self):
        """Test that each generation produces unique keys."""
        key1 = generate_key_pair()
        key2 = generate_key_pair()

        assert key1.public_key != key2.public_key
        assert key1.private_key != key2.private_key
        assert key1.key_id != key2.key_id

    def test_key_id_is_hex_string(self):
        """Test that key_id is a valid hex string."""
        key_pair = generate_key_pair()

        assert len(key_pair.key_id) == 8
        # Should be valid hex
        int(key_pair.key_id, 16)


class TestSigningAndVerification:
    """Tests for message signing and verification."""

    def test_sign_and_verify_roundtrip(self):
        """Test that a signed message can be verified."""
        key_pair = generate_key_pair()
        message = b"Hello, World!"

        signature = sign_message(message, key_pair.private_key)
        result = verify_signature(message, signature, key_pair.public_key)

        assert result is True

    def test_verify_fails_with_wrong_message(self):
        """Test that verification fails if message is tampered."""
        key_pair = generate_key_pair()
        message = b"Original message"
        tampered = b"Tampered message"

        signature = sign_message(message, key_pair.private_key)

        with pytest.raises(SignatureError):
            verify_signature(tampered, signature, key_pair.public_key)

    def test_verify_fails_with_wrong_key(self):
        """Test that verification fails with wrong public key."""
        key1 = generate_key_pair()
        key2 = generate_key_pair()
        message = b"Test message"

        signature = sign_message(message, key1.private_key)

        with pytest.raises(SignatureError):
            verify_signature(message, signature, key2.public_key)

    def test_sign_different_messages_different_signatures(self):
        """Test that different messages produce different signatures."""
        key_pair = generate_key_pair()

        sig1 = sign_message(b"Message 1", key_pair.private_key)
        sig2 = sign_message(b"Message 2", key_pair.private_key)

        assert sig1 != sig2

    def test_sign_same_message_same_signature(self):
        """Test that signing same message produces same signature (deterministic)."""
        key_pair = generate_key_pair()
        message = b"Deterministic test"

        sig1 = sign_message(message, key_pair.private_key)
        sig2 = sign_message(message, key_pair.private_key)

        # Ed25519 is deterministic
        assert sig1 == sig2


class TestKeyManager:
    """Tests for KeyManager class."""

    @pytest.fixture
    def temp_key_dir(self):
        """Create a temporary key directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def key_manager(self, temp_key_dir):
        """Create a KeyManager with temp directory."""
        return KeyManager(agent_id="test-agent", key_dir=temp_key_dir)

    def test_has_key_returns_false_initially(self, key_manager):
        """Test that has_key returns False before generation."""
        assert key_manager.has_key() is False

    def test_generate_creates_key(self, key_manager):
        """Test that generate creates a new key pair."""
        key_pair = key_manager.generate()

        assert key_pair.public_key is not None
        assert key_pair.private_key is not None
        assert key_manager.has_key() is True

    def test_generate_raises_if_exists(self, key_manager):
        """Test that generate raises if key already exists."""
        key_manager.generate()

        with pytest.raises(KeyAlreadyExistsError):
            key_manager.generate()

    def test_generate_with_force_overwrites(self, key_manager):
        """Test that generate with force=True overwrites existing key."""
        original = key_manager.generate()
        new_key = key_manager.generate(force=True)

        assert new_key.public_key != original.public_key

    def test_get_key_pair_returns_stored_key(self, key_manager):
        """Test that get_key_pair returns the stored key."""
        original = key_manager.generate()
        retrieved = key_manager.get_key_pair()

        assert retrieved.public_key == original.public_key
        assert retrieved.private_key == original.private_key

    def test_get_key_pair_raises_if_not_found(self, key_manager):
        """Test that get_key_pair raises if no key exists."""
        with pytest.raises(KeyNotFoundError):
            key_manager.get_key_pair()

    def test_get_public_key_returns_only_public(self, key_manager):
        """Test that get_public_key returns just the public key."""
        key_manager.generate()
        public_key = key_manager.get_public_key()

        assert public_key is not None
        assert isinstance(public_key, str)

    def test_rotate_creates_new_key(self, key_manager):
        """Test that rotate creates a new key pair."""
        original = key_manager.generate()
        new_key, old_key = key_manager.rotate()

        assert new_key.public_key != original.public_key
        assert old_key.public_key == original.public_key

    def test_rotate_archives_old_key(self, key_manager, temp_key_dir):
        """Test that rotate archives the old key."""
        key_manager.generate()
        key_manager.rotate()

        archive_dir = temp_key_dir / "test-agent" / "archive"
        assert archive_dir.exists()
        assert len(list(archive_dir.iterdir())) == 1

    def test_sign_with_manager(self, key_manager):
        """Test signing through KeyManager."""
        key_manager.generate()
        message = b"Test message"

        signature = key_manager.sign(message)

        assert signature is not None
        assert len(signature) > 0

    def test_verify_with_manager(self, key_manager):
        """Test verification through KeyManager."""
        key_manager.generate()
        message = b"Test message"
        signature = key_manager.sign(message)

        result = key_manager.verify(message, signature)

        assert result is True

    def test_delete_removes_key(self, key_manager):
        """Test that delete removes the key."""
        key_manager.generate()
        assert key_manager.has_key() is True

        result = key_manager.delete()

        assert result is True
        assert key_manager.has_key() is False

    def test_delete_returns_false_if_not_found(self, key_manager):
        """Test that delete returns False if no key exists."""
        result = key_manager.delete()
        assert result is False


class TestKeyPairDataclass:
    """Tests for KeyPair dataclass."""

    def test_public_key_bytes_property(self):
        """Test public_key_bytes property."""
        key_pair = generate_key_pair()
        public_bytes = key_pair.public_key_bytes

        assert isinstance(public_bytes, bytes)
        assert len(public_bytes) == 32

    def test_private_key_bytes_property(self):
        """Test private_key_bytes property."""
        key_pair = generate_key_pair()
        private_bytes = key_pair.private_key_bytes

        assert isinstance(private_bytes, bytes)
        assert len(private_bytes) == 32

    def test_private_key_bytes_returns_none_if_missing(self):
        """Test that private_key_bytes returns None if private key not set."""
        key_pair = KeyPair(public_key="dGVzdA==", private_key=None)
        assert key_pair.private_key_bytes is None


class TestFilePermissions:
    """Tests for secure file permissions."""

    def test_private_key_has_restricted_permissions(self):
        """Test that private key file has 600 permissions."""
        import stat

        with tempfile.TemporaryDirectory() as tmpdir:
            key_dir = Path(tmpdir)
            manager = KeyManager(agent_id="perm-test", key_dir=key_dir)
            manager.generate()

            private_path = key_dir / "perm-test" / "private.key"
            mode = private_path.stat().st_mode

            # Check that only owner has read/write
            assert mode & stat.S_IRWXU == stat.S_IRUSR | stat.S_IWUSR
            assert mode & stat.S_IRWXG == 0
            assert mode & stat.S_IRWXO == 0
