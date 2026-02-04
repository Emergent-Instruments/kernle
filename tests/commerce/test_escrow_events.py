"""Tests for escrow events module."""

from kernle.commerce.escrow.events import (
    DeliveredEvent,
    DisputedEvent,
    DisputeResolvedEvent,
    EscrowCreatedEvent,
    EscrowEventIndexer,
    EscrowEventMonitor,
    EscrowEventParser,
    EscrowEventType,
    FundedEvent,
    RefundedEvent,
    ReleasedEvent,
    WorkerAssignedEvent,
)


class TestEventDataClasses:
    """Tests for event data classes."""

    def test_funded_event_creation(self):
        """Test creating a FundedEvent."""
        event = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "a" * 64,
            block_number=12345,
            log_index=0,
            client="0x0987654321098765432109876543210987654321",
            amount=100_000_000,
        )

        assert event.event_type == EscrowEventType.FUNDED
        assert event.client == "0x0987654321098765432109876543210987654321"
        assert event.amount == 100_000_000

    def test_worker_assigned_event(self):
        """Test creating a WorkerAssignedEvent."""
        event = WorkerAssignedEvent(
            event_type=EscrowEventType.WORKER_ASSIGNED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "b" * 64,
            block_number=12346,
            log_index=0,
            worker="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        assert event.event_type == EscrowEventType.WORKER_ASSIGNED
        assert event.worker == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    def test_delivered_event(self):
        """Test creating a DeliveredEvent."""
        event = DeliveredEvent(
            event_type=EscrowEventType.DELIVERED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "c" * 64,
            block_number=12347,
            log_index=0,
            worker="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            deliverable_hash="0x" + "d" * 64,
        )

        assert event.event_type == EscrowEventType.DELIVERED
        assert event.deliverable_hash == "0x" + "d" * 64

    def test_released_event(self):
        """Test creating a ReleasedEvent."""
        event = ReleasedEvent(
            event_type=EscrowEventType.RELEASED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "e" * 64,
            block_number=12348,
            log_index=0,
            worker="0xcccccccccccccccccccccccccccccccccccccccc",
            amount=50_000_000,
        )

        assert event.event_type == EscrowEventType.RELEASED
        assert event.amount == 50_000_000

    def test_refunded_event(self):
        """Test creating a RefundedEvent."""
        event = RefundedEvent(
            event_type=EscrowEventType.REFUNDED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "f" * 64,
            block_number=12349,
            log_index=0,
            client="0xdddddddddddddddddddddddddddddddddddddddd",
            amount=75_000_000,
        )

        assert event.event_type == EscrowEventType.REFUNDED
        assert event.client == "0xdddddddddddddddddddddddddddddddddddddddd"

    def test_disputed_event(self):
        """Test creating a DisputedEvent."""
        event = DisputedEvent(
            event_type=EscrowEventType.DISPUTED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "1" * 64,
            block_number=12350,
            log_index=0,
            disputant="0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        )

        assert event.event_type == EscrowEventType.DISPUTED
        assert event.disputant == "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    def test_dispute_resolved_event(self):
        """Test creating a DisputeResolvedEvent."""
        event = DisputeResolvedEvent(
            event_type=EscrowEventType.DISPUTE_RESOLVED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "2" * 64,
            block_number=12351,
            log_index=0,
            recipient="0xffffffffffffffffffffffffffffffffffffffff",
            amount=100_000_000,
        )

        assert event.event_type == EscrowEventType.DISPUTE_RESOLVED
        assert event.recipient == "0xffffffffffffffffffffffffffffffffffffffff"

    def test_escrow_created_event(self):
        """Test creating an EscrowCreatedEvent."""
        event = EscrowCreatedEvent(
            event_type=EscrowEventType.ESCROW_CREATED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "3" * 64,
            block_number=12352,
            log_index=0,
            job_id="0x" + "4" * 64,
            escrow="0x5555555555555555555555555555555555555555",
            client="0x6666666666666666666666666666666666666666",
            amount=200_000_000,
        )

        assert event.event_type == EscrowEventType.ESCROW_CREATED
        assert event.escrow == "0x5555555555555555555555555555555555555555"


class TestEscrowEventParser:
    """Tests for event parser."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = EscrowEventParser()
        assert parser is not None

    def test_parse_log_returns_none_for_empty(self):
        """Test that empty logs return None."""
        parser = EscrowEventParser()
        result = parser.parse_log({})
        assert result is None

    def test_parse_log_returns_none_for_unknown(self):
        """Test that unknown events return None."""
        parser = EscrowEventParser()
        log = {
            "topics": ["0x" + "0" * 64],  # Unknown topic
            "data": "0x",
            "address": "0x1234567890123456789012345678901234567890",
            "transactionHash": "0x" + "a" * 64,
            "blockNumber": 12345,
            "logIndex": 0,
        }
        result = parser.parse_log(log)
        # Stub returns None for all logs
        assert result is None

    def test_parse_logs_filters_none(self):
        """Test that parse_logs filters out None results."""
        parser = EscrowEventParser()
        logs = [
            {"topics": []},
            {"topics": ["0x" + "0" * 64]},
        ]
        results = parser.parse_logs(logs)
        assert results == []


class TestEscrowEventMonitor:
    """Tests for event monitor."""

    def test_monitor_initialization(self):
        """Test monitor can be initialized."""
        monitor = EscrowEventMonitor(
            rpc_url="https://sepolia.base.org",
            factory_address="0x1234567890123456789012345678901234567890",
        )
        assert monitor.rpc_url == "https://sepolia.base.org"
        assert monitor.factory_address == "0x1234567890123456789012345678901234567890"
        assert not monitor.is_running

    def test_add_handler(self):
        """Test adding event handler."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        events_received = []

        def handler(event):
            events_received.append(event)

        monitor.add_handler(EscrowEventType.FUNDED, handler)
        assert EscrowEventType.FUNDED in monitor._handlers
        assert handler in monitor._handlers[EscrowEventType.FUNDED]

    def test_remove_handler(self):
        """Test removing event handler."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        def handler(event):
            pass

        monitor.add_handler(EscrowEventType.RELEASED, handler)
        result = monitor.remove_handler(EscrowEventType.RELEASED, handler)
        assert result is True
        assert handler not in monitor._handlers.get(EscrowEventType.RELEASED, [])

    def test_remove_nonexistent_handler(self):
        """Test removing handler that doesn't exist."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        def handler(event):
            pass

        result = monitor.remove_handler(EscrowEventType.FUNDED, handler)
        assert result is False

    def test_add_escrow_address(self):
        """Test adding escrow address to monitor."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        monitor.add_escrow("0x1234567890123456789012345678901234567890")
        assert "0x1234567890123456789012345678901234567890" in monitor._escrow_addresses

    def test_add_duplicate_escrow_address(self):
        """Test adding same escrow address twice doesn't duplicate."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        monitor.add_escrow("0x1234567890123456789012345678901234567890")
        monitor.add_escrow("0x1234567890123456789012345678901234567890")
        assert monitor._escrow_addresses.count("0x1234567890123456789012345678901234567890") == 1

    def test_remove_escrow_address(self):
        """Test removing escrow address."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        monitor.add_escrow("0x1234567890123456789012345678901234567890")
        result = monitor.remove_escrow("0x1234567890123456789012345678901234567890")
        assert result is True
        assert "0x1234567890123456789012345678901234567890" not in monitor._escrow_addresses

    def test_start_stop_monitor(self):
        """Test starting and stopping monitor."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        monitor.start()
        assert monitor.is_running

        monitor.stop()
        assert not monitor.is_running

    def test_get_historical_events_returns_empty(self):
        """Test historical events returns empty list (stub)."""
        monitor = EscrowEventMonitor(rpc_url="https://sepolia.base.org")

        events = monitor.get_historical_events(
            contract_address="0x1234567890123456789012345678901234567890",
            from_block=0,
        )
        assert events == []


class TestEscrowEventIndexer:
    """Tests for event indexer."""

    def test_indexer_initialization(self):
        """Test indexer can be initialized."""
        indexer = EscrowEventIndexer()
        assert len(indexer._events) == 0

    def test_index_event(self):
        """Test indexing an event."""
        indexer = EscrowEventIndexer()

        event = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "a" * 64,
            block_number=12345,
            log_index=0,
            client="0x0987654321098765432109876543210987654321",
            amount=100_000_000,
        )

        indexer.index(event)

        assert len(indexer._events) == 1

    def test_get_events_for_escrow(self):
        """Test getting events for specific escrow."""
        indexer = EscrowEventIndexer()

        event1 = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x1111111111111111111111111111111111111111",
            tx_hash="0x" + "a" * 64,
            block_number=1,
            log_index=0,
        )
        event2 = ReleasedEvent(
            event_type=EscrowEventType.RELEASED,
            contract_address="0x1111111111111111111111111111111111111111",
            tx_hash="0x" + "b" * 64,
            block_number=2,
            log_index=0,
        )
        event3 = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x2222222222222222222222222222222222222222",
            tx_hash="0x" + "c" * 64,
            block_number=3,
            log_index=0,
        )

        indexer.index(event1)
        indexer.index(event2)
        indexer.index(event3)

        escrow1_events = indexer.get_events_for_escrow("0x1111111111111111111111111111111111111111")
        assert len(escrow1_events) == 2

        escrow2_events = indexer.get_events_for_escrow("0x2222222222222222222222222222222222222222")
        assert len(escrow2_events) == 1

    def test_get_events_by_type(self):
        """Test getting events by type."""
        indexer = EscrowEventIndexer()

        event1 = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "a" * 64,
            block_number=1,
            log_index=0,
        )
        event2 = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x0987654321098765432109876543210987654321",
            tx_hash="0x" + "b" * 64,
            block_number=2,
            log_index=0,
        )
        event3 = ReleasedEvent(
            event_type=EscrowEventType.RELEASED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "c" * 64,
            block_number=3,
            log_index=0,
        )

        indexer.index(event1)
        indexer.index(event2)
        indexer.index(event3)

        funded_events = indexer.get_events_by_type(EscrowEventType.FUNDED)
        assert len(funded_events) == 2

        released_events = indexer.get_events_by_type(EscrowEventType.RELEASED)
        assert len(released_events) == 1

    def test_get_recent_events(self):
        """Test getting recent events."""
        indexer = EscrowEventIndexer()

        for i in range(5):
            event = FundedEvent(
                event_type=EscrowEventType.FUNDED,
                contract_address=f"0x{str(i) * 40}",
                tx_hash=f"0x{str(i) * 64}",
                block_number=i,
                log_index=0,
            )
            indexer.index(event)

        recent = indexer.get_recent_events(limit=3)
        assert len(recent) == 3
        # Should be in reverse order (newest first)
        assert recent[0].block_number == 4
        assert recent[1].block_number == 3
        assert recent[2].block_number == 2

    def test_clear_indexer(self):
        """Test clearing the indexer."""
        indexer = EscrowEventIndexer()

        event = FundedEvent(
            event_type=EscrowEventType.FUNDED,
            contract_address="0x1234567890123456789012345678901234567890",
            tx_hash="0x" + "a" * 64,
            block_number=1,
            log_index=0,
        )
        indexer.index(event)

        indexer.clear()

        assert len(indexer._events) == 0
        assert len(indexer._by_escrow) == 0
        assert len(indexer._by_type) == 0
