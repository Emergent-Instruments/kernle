"""Tests for agent messaging module."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from kernle.comms.messaging import (
    Message,
    MessagePriority,
    MessageStatus,
    MessageStore,
    Messenger,
    MessageNotFoundError,
)
from kernle.storage import SQLiteStorage


class TestMessage:
    """Tests for Message dataclass."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all message fields."""
        msg = Message(
            id=uuid4(),
            from_agent="alice",
            to_agent="bob",
            body="Hello!",
            subject="Greeting",
            priority=MessagePriority.HIGH,
        )
        d = msg.to_dict()
        
        assert d["from_agent"] == "alice"
        assert d["to_agent"] == "bob"
        assert d["body"] == "Hello!"
        assert d["subject"] == "Greeting"
        assert d["priority"] == "high"
        assert d["status"] == "pending"

    def test_from_dict_creates_message(self):
        """Test that from_dict correctly reconstructs a message."""
        msg_id = uuid4()
        data = {
            "id": str(msg_id),
            "from_agent": "alice",
            "to_agent": "bob",
            "body": "Test message",
            "subject": None,
            "reply_to_id": None,
            "priority": "normal",
            "status": "pending",
            "created_at": "2026-02-04T12:00:00+00:00",
            "delivered_at": None,
            "read_at": None,
            "expires_at": None,
            "metadata": {},
        }
        
        msg = Message.from_dict(data)
        assert msg.id == msg_id
        assert msg.from_agent == "alice"
        assert msg.to_agent == "bob"
        assert msg.body == "Test message"

    def test_is_expired_false_when_no_expiry(self):
        """Test that is_expired is False when no expiry set."""
        msg = Message(
            id=uuid4(),
            from_agent="alice",
            to_agent="bob",
            body="No expiry",
        )
        assert not msg.is_expired

    def test_is_expired_true_when_past_expiry(self):
        """Test that is_expired is True when past expiry time."""
        msg = Message(
            id=uuid4(),
            from_agent="alice",
            to_agent="bob",
            body="Expired",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert msg.is_expired

    def test_is_expired_false_when_before_expiry(self):
        """Test that is_expired is False when before expiry time."""
        msg = Message(
            id=uuid4(),
            from_agent="alice",
            to_agent="bob",
            body="Not expired",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert not msg.is_expired

    def test_thread_id_is_self_when_root(self):
        """Test that thread_id is message id when no reply_to."""
        msg = Message(
            id=uuid4(),
            from_agent="alice",
            to_agent="bob",
            body="Root message",
        )
        assert msg.thread_id == msg.id

    def test_thread_id_is_reply_to_when_reply(self):
        """Test that thread_id is reply_to_id when set."""
        root_id = uuid4()
        msg = Message(
            id=uuid4(),
            from_agent="bob",
            to_agent="alice",
            body="Reply",
            reply_to_id=root_id,
        )
        assert msg.thread_id == root_id


class TestMessageStore:
    """Tests for MessageStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        path = Path(tempfile.mktemp(suffix=".db"))
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a SQLiteStorage instance for testing."""
        storage = SQLiteStorage(agent_id="alice", db_path=temp_db)
        yield storage
        storage.close()

    @pytest.fixture
    def message_store(self, storage):
        """Create a message store for agent 'alice'."""
        return MessageStore(storage, "alice")

    def test_save_and_get_message(self, message_store):
        """Test saving and retrieving a message."""
        msg = Message(
            id=uuid4(),
            from_agent="bob",
            to_agent="alice",
            body="Hello Alice!",
            subject="Greeting",
        )
        
        message_store.save(msg)
        retrieved = message_store.get(msg.id)
        
        assert retrieved is not None
        assert retrieved.id == msg.id
        assert retrieved.body == "Hello Alice!"
        assert retrieved.from_agent == "bob"

    def test_get_returns_none_for_missing(self, message_store):
        """Test that get returns None for missing message."""
        result = message_store.get(uuid4())
        assert result is None

    def test_list_inbox_returns_messages_to_agent(self, message_store):
        """Test that list_inbox returns messages addressed to agent."""
        # Message to alice (should appear)
        msg1 = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="To Alice")
        # Message from alice (should not appear)
        msg2 = Message(id=uuid4(), from_agent="alice", to_agent="bob", body="From Alice")
        
        message_store.save(msg1)
        message_store.save(msg2)
        
        inbox = message_store.list_inbox()
        assert len(inbox) == 1
        assert inbox[0].id == msg1.id

    def test_list_inbox_unread_only(self, message_store):
        """Test that unread_only filters read messages."""
        msg1 = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="Unread")
        msg2 = Message(
            id=uuid4(),
            from_agent="bob",
            to_agent="alice",
            body="Read",
            status=MessageStatus.READ,
        )
        
        message_store.save(msg1)
        message_store.save(msg2)
        
        unread = message_store.list_inbox(unread_only=True)
        assert len(unread) == 1
        assert unread[0].body == "Unread"

    def test_list_outbox_returns_sent_messages(self, message_store):
        """Test that list_outbox returns messages from agent."""
        msg = Message(id=uuid4(), from_agent="alice", to_agent="bob", body="Sent")
        message_store.save(msg)
        
        outbox = message_store.list_outbox()
        assert len(outbox) == 1
        assert outbox[0].body == "Sent"

    def test_mark_read_updates_status(self, message_store):
        """Test that mark_read updates message status."""
        msg = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="Test")
        message_store.save(msg)
        
        message_store.mark_read(msg.id)
        
        updated = message_store.get(msg.id)
        assert updated.status == MessageStatus.READ
        assert updated.read_at is not None

    def test_count_unread(self, message_store):
        """Test counting unread messages."""
        msg1 = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="Unread 1")
        msg2 = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="Unread 2")
        msg3 = Message(
            id=uuid4(),
            from_agent="bob",
            to_agent="alice",
            body="Read",
            status=MessageStatus.READ,
        )
        
        message_store.save(msg1)
        message_store.save(msg2)
        message_store.save(msg3)
        
        assert message_store.count_unread() == 2

    def test_delete_removes_message(self, message_store):
        """Test that delete removes a message."""
        msg = Message(id=uuid4(), from_agent="bob", to_agent="alice", body="Delete me")
        message_store.save(msg)
        
        message_store.delete(msg.id)
        
        assert message_store.get(msg.id) is None


class TestMessenger:
    """Tests for high-level Messenger interface."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        path = Path(tempfile.mktemp(suffix=".db"))
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a SQLiteStorage instance for testing."""
        storage = SQLiteStorage(agent_id="test", db_path=temp_db)
        yield storage
        storage.close()

    @pytest.fixture
    def alice(self, storage):
        """Create messenger for alice."""
        return Messenger(storage, "alice")

    @pytest.fixture
    def bob(self, storage):
        """Create messenger for bob."""
        return Messenger(storage, "bob")

    def test_send_creates_message(self, alice):
        """Test that send creates a message."""
        msg = alice.send(to="bob", body="Hello Bob!")
        
        assert msg.from_agent == "alice"
        assert msg.to_agent == "bob"
        assert msg.body == "Hello Bob!"
        assert msg.status == MessageStatus.PENDING

    def test_send_with_subject_and_priority(self, alice):
        """Test send with optional parameters."""
        msg = alice.send(
            to="bob",
            body="Urgent!",
            subject="Important",
            priority=MessagePriority.URGENT,
        )
        
        assert msg.subject == "Important"
        assert msg.priority == MessagePriority.URGENT

    def test_send_with_ttl_sets_expiry(self, alice):
        """Test that ttl_hours sets expiration."""
        msg = alice.send(to="bob", body="Expires soon", ttl_hours=24)
        
        assert msg.expires_at is not None
        expected = datetime.now(timezone.utc) + timedelta(hours=24)
        assert abs((msg.expires_at - expected).total_seconds()) < 5

    def test_inbox_shows_received_messages(self, alice, bob):
        """Test that inbox shows messages to agent."""
        bob.send(to="alice", body="Message 1")
        bob.send(to="alice", body="Message 2")
        
        inbox = alice.inbox()
        assert len(inbox) == 2

    def test_outbox_shows_sent_messages(self, alice):
        """Test that outbox shows messages from agent."""
        alice.send(to="bob", body="Sent 1")
        alice.send(to="charlie", body="Sent 2")
        
        outbox = alice.outbox()
        assert len(outbox) == 2

    def test_read_marks_message_as_read(self, alice, bob):
        """Test that read marks message as read."""
        msg = bob.send(to="alice", body="Read me")
        
        read_msg = alice.read(msg.id)
        
        assert read_msg.status == MessageStatus.READ
        assert read_msg.read_at is not None

    def test_read_raises_for_missing_message(self, alice):
        """Test that read raises for non-existent message."""
        with pytest.raises(MessageNotFoundError):
            alice.read(uuid4())

    def test_reply_creates_threaded_message(self, alice, bob):
        """Test that reply creates a message in the thread."""
        original = bob.send(to="alice", body="Original message", subject="Topic")
        
        reply = alice.reply(original.id, body="My reply")
        
        assert reply.reply_to_id == original.id
        assert reply.to_agent == "bob"
        assert reply.from_agent == "alice"
        assert reply.subject == "Re: Topic"

    def test_thread_returns_all_messages(self, alice, bob):
        """Test that thread returns all messages in conversation."""
        msg1 = bob.send(to="alice", body="First")
        msg2 = alice.reply(msg1.id, body="Second")
        msg3 = bob.reply(msg2.id, body="Third")
        
        thread = alice.thread(msg1.id)
        
        assert len(thread) == 3
        # Should be in chronological order
        assert thread[0].body == "First"
        assert thread[1].body == "Second"
        assert thread[2].body == "Third"

    def test_unread_count(self, alice, bob):
        """Test unread message count."""
        bob.send(to="alice", body="Unread 1")
        bob.send(to="alice", body="Unread 2")
        
        assert alice.unread_count() == 2
        
        # Read one message
        inbox = alice.inbox()
        alice.read(inbox[0].id)
        
        assert alice.unread_count() == 1

    def test_delete_message(self, alice, bob):
        """Test deleting a message."""
        msg = bob.send(to="alice", body="Delete me")
        
        alice.delete(msg.id)
        
        # Message should be gone from inbox
        inbox = alice.inbox()
        assert len(inbox) == 0
