"""
Cryptographic utilities for Kernle Comms.

Provides Ed25519 key management for agent identity:
- Key pair generation
- Message signing and verification
- Key storage and rotation
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default key storage directory
DEFAULT_KEY_DIR = Path.home() / ".kernle" / "keys"


class CryptoError(Exception):
    """Base exception for crypto errors."""

    pass


class KeyNotFoundError(CryptoError):
    """Private key not found."""

    pass


class SignatureError(CryptoError):
    """Signature verification failed."""

    pass


class KeyAlreadyExistsError(CryptoError):
    """Key already exists (use rotate to replace)."""

    pass


@dataclass
class KeyPair:
    """Ed25519 key pair.

    Attributes:
        public_key: Base64-encoded public key
        private_key: Base64-encoded private key (optional, for security)
        created_at: When the key was generated
        key_id: Short identifier derived from public key
    """

    public_key: str
    private_key: Optional[str] = None
    created_at: Optional[datetime] = None
    key_id: Optional[str] = None

    @property
    def public_key_bytes(self) -> bytes:
        """Get public key as bytes."""
        return base64.b64decode(self.public_key)

    @property
    def private_key_bytes(self) -> Optional[bytes]:
        """Get private key as bytes."""
        if self.private_key is None:
            return None
        return base64.b64decode(self.private_key)


def generate_key_pair() -> KeyPair:
    """Generate a new Ed25519 key pair.

    Returns:
        KeyPair with public and private keys

    Raises:
        CryptoError: If key generation fails
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Serialize to raw bytes then base64
        private_bytes = private_key.private_bytes_raw()
        public_bytes = public_key.public_bytes_raw()

        public_b64 = base64.b64encode(public_bytes).decode("ascii")
        private_b64 = base64.b64encode(private_bytes).decode("ascii")

        # Generate key_id from first 8 chars of public key hash
        import hashlib

        key_id = hashlib.sha256(public_bytes).hexdigest()[:8]

        return KeyPair(
            public_key=public_b64,
            private_key=private_b64,
            created_at=datetime.now(timezone.utc),
            key_id=key_id,
        )
    except ImportError:
        raise CryptoError(
            "cryptography package not installed. Install with: pip install cryptography"
        )
    except Exception as e:
        logger.error(f"Key generation failed: {e}")
        raise CryptoError(f"Failed to generate key pair: {e}") from e


def sign_message(message: bytes, private_key_b64: str) -> str:
    """Sign a message with Ed25519 private key.

    Args:
        message: Message bytes to sign
        private_key_b64: Base64-encoded private key

    Returns:
        Base64-encoded signature

    Raises:
        CryptoError: If signing fails
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_bytes = base64.b64decode(private_key_b64)
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)

        signature = private_key.sign(message)
        return base64.b64encode(signature).decode("ascii")
    except ImportError:
        raise CryptoError("cryptography package not installed")
    except Exception as e:
        logger.error(f"Signing failed: {e}")
        raise CryptoError(f"Failed to sign message: {e}") from e


def verify_signature(message: bytes, signature_b64: str, public_key_b64: str) -> bool:
    """Verify a message signature with Ed25519 public key.

    Args:
        message: Original message bytes
        signature_b64: Base64-encoded signature
        public_key_b64: Base64-encoded public key

    Returns:
        True if signature is valid, False otherwise

    Raises:
        SignatureError: If signature is invalid
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_bytes = base64.b64decode(public_key_b64)
        signature = base64.b64decode(signature_b64)

        public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
        public_key.verify(signature, message)
        return True
    except ImportError:
        raise CryptoError("cryptography package not installed")
    except Exception as e:
        logger.debug(f"Signature verification failed: {e}")
        raise SignatureError(f"Invalid signature: {e}") from e


class KeyManager:
    """Manages agent key pairs for signing and verification.

    Keys are stored locally in the key directory with the following structure:
    - {key_dir}/{agent_id}/private.key - Private key (base64)
    - {key_dir}/{agent_id}/public.key - Public key (base64)
    - {key_dir}/{agent_id}/meta.json - Key metadata
    """

    def __init__(self, agent_id: str, key_dir: Optional[Path] = None):
        """Initialize key manager.

        Args:
            agent_id: Agent identifier
            key_dir: Directory to store keys (default: ~/.kernle/keys)
        """
        self.agent_id = agent_id
        self.key_dir = key_dir or DEFAULT_KEY_DIR
        self._agent_key_dir = self.key_dir / agent_id

    def has_key(self) -> bool:
        """Check if agent has a key pair."""
        return (self._agent_key_dir / "private.key").exists()

    def generate(self, force: bool = False) -> KeyPair:
        """Generate and store a new key pair.

        Args:
            force: If True, overwrite existing key

        Returns:
            The generated KeyPair

        Raises:
            KeyAlreadyExistsError: If key exists and force=False
        """
        if self.has_key() and not force:
            raise KeyAlreadyExistsError(
                f"Key already exists for '{self.agent_id}'. Use force=True or rotate() to replace."
            )

        key_pair = generate_key_pair()
        self._save_key_pair(key_pair)
        logger.info(f"Generated new key pair for agent: {self.agent_id}")
        return key_pair

    def rotate(self) -> Tuple[KeyPair, Optional[KeyPair]]:
        """Rotate key pair, keeping old key for reference.

        Returns:
            Tuple of (new_key, old_key or None)
        """
        old_key = None
        if self.has_key():
            old_key = self.get_key_pair()
            # Archive old key
            self._archive_key()

        new_key = generate_key_pair()
        self._save_key_pair(new_key)
        logger.info(f"Rotated key pair for agent: {self.agent_id}")
        return new_key, old_key

    def get_key_pair(self) -> KeyPair:
        """Get the agent's key pair.

        Returns:
            KeyPair with both public and private keys

        Raises:
            KeyNotFoundError: If no key exists
        """
        if not self.has_key():
            raise KeyNotFoundError(f"No key found for agent '{self.agent_id}'")

        private_path = self._agent_key_dir / "private.key"
        public_path = self._agent_key_dir / "public.key"
        meta_path = self._agent_key_dir / "meta.json"

        private_key = private_path.read_text().strip()
        public_key = public_path.read_text().strip()

        created_at = None
        key_id = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if "created_at" in meta:
                created_at = datetime.fromisoformat(meta["created_at"])
            key_id = meta.get("key_id")

        return KeyPair(
            public_key=public_key,
            private_key=private_key,
            created_at=created_at,
            key_id=key_id,
        )

    def get_public_key(self) -> str:
        """Get just the public key (safe to share).

        Returns:
            Base64-encoded public key

        Raises:
            KeyNotFoundError: If no key exists
        """
        if not self.has_key():
            raise KeyNotFoundError(f"No key found for agent '{self.agent_id}'")

        public_path = self._agent_key_dir / "public.key"
        return public_path.read_text().strip()

    def sign(self, message: bytes) -> str:
        """Sign a message with the agent's private key.

        Args:
            message: Message bytes to sign

        Returns:
            Base64-encoded signature

        Raises:
            KeyNotFoundError: If no key exists
        """
        key_pair = self.get_key_pair()
        if key_pair.private_key is None:
            raise KeyNotFoundError("Private key not available")
        return sign_message(message, key_pair.private_key)

    def verify(self, message: bytes, signature: str, public_key: Optional[str] = None) -> bool:
        """Verify a message signature.

        Args:
            message: Original message bytes
            signature: Base64-encoded signature
            public_key: Public key to verify with (default: this agent's key)

        Returns:
            True if valid

        Raises:
            SignatureError: If signature is invalid
        """
        if public_key is None:
            public_key = self.get_public_key()
        return verify_signature(message, signature, public_key)

    def delete(self) -> bool:
        """Delete the agent's key pair.

        Returns:
            True if deleted, False if not found
        """
        if not self.has_key():
            return False

        import shutil

        shutil.rmtree(self._agent_key_dir)
        logger.info(f"Deleted key pair for agent: {self.agent_id}")
        return True

    def _save_key_pair(self, key_pair: KeyPair) -> None:
        """Save key pair to disk."""
        self._agent_key_dir.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions on private key
        private_path = self._agent_key_dir / "private.key"
        private_path.write_text(key_pair.private_key)
        os.chmod(private_path, 0o600)

        public_path = self._agent_key_dir / "public.key"
        public_path.write_text(key_pair.public_key)

        meta_path = self._agent_key_dir / "meta.json"
        meta = {
            "created_at": key_pair.created_at.isoformat() if key_pair.created_at else None,
            "key_id": key_pair.key_id,
            "agent_id": self.agent_id,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    def _archive_key(self) -> None:
        """Archive current key before rotation."""
        if not self.has_key():
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_dir = self._agent_key_dir / "archive" / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        for file in ["private.key", "public.key", "meta.json"]:
            src = self._agent_key_dir / file
            if src.exists():
                shutil.copy2(src, archive_dir / file)
