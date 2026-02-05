"""
Agent messaging module for Kernle Comms.

Provides asynchronous message passing between agents with threading,
priority levels, and expiration support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class MessageStatus(str, Enum):
    """Status of an agent message."""
    PENDING = "pending"      # Created, not yet delivered
    DELIVERED = "delivered"  # Delivered to recipient's inbox
    READ = "read"           # Recipient has read the message
    EXPIRED = "expired"     # TTL exceeded, message expired
    FAILED = "failed"       # Delivery failed


class MessagePriority(str, Enum):
    """Priority level for messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessagingError(Exception):
    """Base exception for messaging operations."""
    pass


class MessageNotFoundError(MessagingError):
    """Raised when a message is not found."""
    pass


class DeliveryError(MessagingError):
    """Raised when message delivery fails."""
    pass


@dataclass
class Message:
    """Represents a message between agents."""
    id: UUID
    from_agent: str
    to_agent: str
    body: str
    subject: Optional[str] = None
    reply_to_id: Optional[UUID] = None
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        return {
            "id": str(self.id),
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "subject": self.subject,
            "body": self.body,
            "reply_to_id": str(self.reply_to_id) if self.reply_to_id else None,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            id=UUID(data["id"]),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            subject=data.get("subject"),
            body=data["body"],
            reply_to_id=UUID(data["reply_to_id"]) if data.get("reply_to_id") else None,
            priority=MessagePriority(data.get("priority", "normal")),
            status=MessageStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]),
            delivered_at=datetime.fromisoformat(data["delivered_at"]) if data.get("delivered_at") else None,
            read_at=datetime.fromisoformat(data["read_at"]) if data.get("read_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_read(self) -> bool:
        """Check if message has been read."""
        return self.status == MessageStatus.READ

    @property
    def thread_id(self) -> UUID:
        """Get the thread ID (root message ID or self if root)."""
        return self.reply_to_id or self.id


class MessageStore:
    """
    Storage for agent messages.
    
    Uses SQLiteStorage or PostgresStorage backend for persistence.
    """

    def __init__(self, storage: Any, agent_id: str):
        """
        Initialize message store.
        
        Args:
            storage: SQLiteStorage or PostgresStorage instance
            agent_id: Current agent's ID (for inbox/outbox queries)
        """
        self._storage = storage
        self.agent_id = agent_id
        self._table = "agent_messages"
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure message table exists."""
        conn = self._storage._get_conn()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    from_agent TEXT NOT NULL,
                    to_agent TEXT NOT NULL,
                    subject TEXT,
                    body TEXT NOT NULL,
                    reply_to_id TEXT,
                    priority TEXT DEFAULT 'normal',
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    read_at TEXT,
                    expires_at TEXT,
                    metadata TEXT DEFAULT '{{}}'
                )
            """)
            # Create indexes for common queries
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_to_agent 
                ON {self._table}(to_agent, status)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_from_agent 
                ON {self._table}(from_agent, created_at)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_thread 
                ON {self._table}(reply_to_id)
            """)
            conn.commit()
        finally:
            conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        """Execute SQL and commit."""
        conn = self._storage._get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: tuple = ()):
        """Execute SQL and fetch one row."""
        conn = self._storage._get_conn()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def _fetchall(self, sql: str, params: tuple = ()) -> list:
        """Execute SQL and fetch all rows."""
        conn = self._storage._get_conn()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def save(self, message: Message) -> Message:
        """Save a message to the store."""
        import json
        self._execute(
            f"""
            INSERT OR REPLACE INTO {self._table} 
            (id, from_agent, to_agent, subject, body, reply_to_id, priority, 
             status, created_at, delivered_at, read_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(message.id),
                message.from_agent,
                message.to_agent,
                message.subject,
                message.body,
                str(message.reply_to_id) if message.reply_to_id else None,
                message.priority.value,
                message.status.value,
                message.created_at.isoformat(),
                message.delivered_at.isoformat() if message.delivered_at else None,
                message.read_at.isoformat() if message.read_at else None,
                message.expires_at.isoformat() if message.expires_at else None,
                json.dumps(message.metadata),
            ),
        )
        return message

    def get(self, message_id: UUID) -> Optional[Message]:
        """Get a message by ID."""
        import json
        row = self._fetchone(
            f"SELECT * FROM {self._table} WHERE id = ?",
            (str(message_id),),
        )
        if not row:
            return None
        return self._row_to_message(row)

    def _row_to_message(self, row) -> Message:
        """Convert database row to Message object."""
        import json
        return Message(
            id=UUID(row[0]),
            from_agent=row[1],
            to_agent=row[2],
            subject=row[3],
            body=row[4],
            reply_to_id=UUID(row[5]) if row[5] else None,
            priority=MessagePriority(row[6]),
            status=MessageStatus(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            delivered_at=datetime.fromisoformat(row[9]) if row[9] else None,
            read_at=datetime.fromisoformat(row[10]) if row[10] else None,
            expires_at=datetime.fromisoformat(row[11]) if row[11] else None,
            metadata=json.loads(row[12]) if row[12] else {},
        )

    def list_inbox(
        self,
        unread_only: bool = False,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[Message]:
        """
        List messages in inbox (messages to this agent).
        
        Args:
            unread_only: Only return unread messages
            limit: Maximum number of messages to return
            include_expired: Include expired messages
        """
        query = f"SELECT * FROM {self._table} WHERE to_agent = ?"
        params: list = [self.agent_id]
        
        if unread_only:
            query += " AND status != 'read'"
        if not include_expired:
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.now(timezone.utc).isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self._fetchall(query, tuple(params))
        return [self._row_to_message(row) for row in rows]

    def list_outbox(self, limit: int = 50) -> list[Message]:
        """List messages sent by this agent."""
        rows = self._fetchall(
            f"""
            SELECT * FROM {self._table} 
            WHERE from_agent = ? 
            ORDER BY created_at DESC 
            LIMIT ?
            """,
            (self.agent_id, limit),
        )
        return [self._row_to_message(row) for row in rows]

    def list_thread(self, thread_id: UUID) -> list[Message]:
        """List all messages in a thread."""
        rows = self._fetchall(
            f"""
            SELECT * FROM {self._table} 
            WHERE id = ? OR reply_to_id = ?
            ORDER BY created_at ASC
            """,
            (str(thread_id), str(thread_id)),
        )
        return [self._row_to_message(row) for row in rows]

    def mark_read(self, message_id: UUID) -> bool:
        """Mark a message as read."""
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            f"""
            UPDATE {self._table} 
            SET status = 'read', read_at = ?
            WHERE id = ? AND to_agent = ?
            """,
            (now, str(message_id), self.agent_id),
        )
        return True

    def mark_delivered(self, message_id: UUID) -> bool:
        """Mark a message as delivered."""
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            f"""
            UPDATE {self._table} 
            SET status = 'delivered', delivered_at = ?
            WHERE id = ?
            """,
            (now, str(message_id)),
        )
        return True

    def delete(self, message_id: UUID) -> bool:
        """Delete a message."""
        self._execute(
            f"DELETE FROM {self._table} WHERE id = ? AND (from_agent = ? OR to_agent = ?)",
            (str(message_id), self.agent_id, self.agent_id),
        )
        return True

    def count_unread(self) -> int:
        """Count unread messages in inbox."""
        row = self._fetchone(
            f"""
            SELECT COUNT(*) FROM {self._table} 
            WHERE to_agent = ? AND status != 'read'
            AND (expires_at IS NULL OR expires_at > ?)
            """,
            (self.agent_id, datetime.now(timezone.utc).isoformat()),
        )
        return row[0] if row else 0


class Messenger:
    """
    High-level messaging interface for agents.
    
    Provides send/receive/reply operations with automatic
    delivery tracking and thread management.
    """

    def __init__(self, storage: Any, agent_id: str):
        """
        Initialize messenger.
        
        Args:
            storage: SQLiteStorage or PostgresStorage instance
            agent_id: Current agent's ID
        """
        self.agent_id = agent_id
        self.message_store = MessageStore(storage, agent_id)

    def send(
        self,
        to: str,
        body: str,
        subject: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_hours: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Message:
        """
        Send a message to another agent.
        
        Args:
            to: Recipient agent ID
            body: Message content
            subject: Optional subject line
            priority: Message priority level
            ttl_hours: Hours until message expires (None = no expiration)
            metadata: Optional metadata dictionary
            
        Returns:
            The created message
        """
        expires_at = None
        if ttl_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        
        message = Message(
            id=uuid4(),
            from_agent=self.agent_id,
            to_agent=to,
            body=body,
            subject=subject,
            priority=priority,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        return self.message_store.save(message)

    def reply(
        self,
        message_id: UUID,
        body: str,
        priority: Optional[MessagePriority] = None,
    ) -> Message:
        """
        Reply to a message.
        
        Args:
            message_id: ID of message to reply to
            body: Reply content
            priority: Priority (defaults to original message's priority)
            
        Returns:
            The reply message
        """
        original = self.message_store.get(message_id)
        if not original:
            raise MessageNotFoundError(f"Message {message_id} not found")
        
        # Reply to the sender
        reply = Message(
            id=uuid4(),
            from_agent=self.agent_id,
            to_agent=original.from_agent,
            body=body,
            subject=f"Re: {original.subject}" if original.subject else None,
            reply_to_id=original.thread_id,  # Thread to root message
            priority=priority or original.priority,
        )
        
        return self.message_store.save(reply)

    def inbox(
        self,
        unread_only: bool = False,
        limit: int = 50,
    ) -> list[Message]:
        """
        Get inbox messages.
        
        Args:
            unread_only: Only return unread messages
            limit: Maximum messages to return
            
        Returns:
            List of messages
        """
        return self.message_store.list_inbox(unread_only=unread_only, limit=limit)

    def outbox(self, limit: int = 50) -> list[Message]:
        """Get sent messages."""
        return self.message_store.list_outbox(limit=limit)

    def read(self, message_id: UUID) -> Message:
        """
        Read a message (marks it as read).
        
        Args:
            message_id: Message ID to read
            
        Returns:
            The message
            
        Raises:
            MessageNotFoundError: If message not found
        """
        message = self.message_store.get(message_id)
        if not message:
            raise MessageNotFoundError(f"Message {message_id} not found")
        
        # Only mark as read if we're the recipient
        if message.to_agent == self.agent_id and not message.is_read:
            self.message_store.mark_read(message_id)
            message.status = MessageStatus.READ
            message.read_at = datetime.now(timezone.utc)
        
        return message

    def thread(self, message_id: UUID) -> list[Message]:
        """
        Get all messages in a thread.
        
        Args:
            message_id: Any message ID in the thread
            
        Returns:
            List of messages in thread, chronologically ordered
        """
        message = self.message_store.get(message_id)
        if not message:
            raise MessageNotFoundError(f"Message {message_id} not found")
        
        return self.message_store.list_thread(message.thread_id)

    def unread_count(self) -> int:
        """Get count of unread messages."""
        return self.message_store.count_unread()

    def delete(self, message_id: UUID) -> bool:
        """Delete a message."""
        return self.message_store.delete(message_id)
