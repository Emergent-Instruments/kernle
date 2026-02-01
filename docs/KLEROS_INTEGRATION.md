# Kleros Integration Design Document

> **Status:** Draft  
> **Author:** Claire (AI Agent)  
> **Created:** 2026-01-31  
> **Last Updated:** 2026-01-31

## Executive Summary

This document outlines the integration of Kleros decentralized arbitration into the Kernle Commerce escrow system. By implementing the ERC-792 Arbitrable interface and ERC-1497 Evidence Standard, our escrow contracts will support trustless, decentralized dispute resolution for agent-to-agent commerce.

**Key Benefits:**
- Decentralized, trustless dispute resolution
- No single point of failure or bias
- Automatic enforcement of rulings via smart contracts
- Compatible with any ERC-792 compliant arbitrator (upgradeable)

---

## Table of Contents

1. [How ERC-792 Arbitrable Works](#1-how-erc-792-arbitrable-works)
2. [Contract Changes Needed](#2-contract-changes-needed)
3. [Dispute Flow](#3-dispute-flow)
4. [Cost Considerations](#4-cost-considerations)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Security Considerations](#6-security-considerations)
7. [Appendix](#appendix)

---

## 1. How ERC-792 Arbitrable Works

### 1.1 Overview

ERC-792 defines a standard interface for decentralized arbitration with two key components:

| Component | Role |
|-----------|------|
| **Arbitrable** | Contract that can have disputes (our escrow) |
| **Arbitrator** | Contract that resolves disputes (Kleros Court) |

This separation allows any Arbitrable contract to work with any Arbitrator contract, enabling composability and upgradability.

### 1.2 IArbitrable Interface

Our escrow must implement the `IArbitrable` interface:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./IArbitrator.sol";

interface IArbitrable {
    /**
     * @dev Emitted when a ruling is given.
     * @param _arbitrator The arbitrator giving the ruling.
     * @param _disputeID ID of the dispute in the Arbitrator contract.
     * @param _ruling The ruling which was given.
     */
    event Ruling(
        IArbitrator indexed _arbitrator, 
        uint256 indexed _disputeID, 
        uint256 _ruling
    );

    /**
     * @dev Called by the arbitrator to enforce a ruling.
     * @param _disputeID ID of the dispute in the Arbitrator contract.
     * @param _ruling Ruling given by the arbitrator (0 = refused to rule).
     */
    function rule(uint256 _disputeID, uint256 _ruling) external;
}
```

### 1.3 IArbitrator Interface

The Arbitrator (Kleros Court) provides these key functions:

```solidity
interface IArbitrator {
    enum DisputeStatus { Waiting, Appealable, Solved }

    // Events
    event DisputeCreation(uint256 indexed _disputeID, IArbitrable indexed _arbitrable);
    event AppealPossible(uint256 indexed _disputeID, IArbitrable indexed _arbitrable);
    event AppealDecision(uint256 indexed _disputeID, IArbitrable indexed _arbitrable);

    // Create a dispute (called by Arbitrable, paid with ETH)
    function createDispute(
        uint256 _choices, 
        bytes calldata _extraData
    ) external payable returns (uint256 disputeID);

    // Get cost to create a dispute
    function arbitrationCost(bytes calldata _extraData) external view returns (uint256 cost);

    // Appeal a ruling
    function appeal(uint256 _disputeID, bytes calldata _extraData) external payable;

    // Get cost to appeal
    function appealCost(
        uint256 _disputeID, 
        bytes calldata _extraData
    ) external view returns (uint256 cost);

    // Get appeal window
    function appealPeriod(uint256 _disputeID) external view returns (uint256 start, uint256 end);

    // Check dispute status
    function disputeStatus(uint256 _disputeID) external view returns (DisputeStatus status);

    // Get current/pending ruling
    function currentRuling(uint256 _disputeID) external view returns (uint256 ruling);
}
```

### 1.4 ERC-1497 Evidence Standard

ERC-1497 standardizes how evidence is submitted and displayed:

- **MetaEvidence**: Context about the dispute (submitted at contract creation)
  - Description of the agreement
  - Ruling options and their meanings
  - Reference to original contract/terms
  
- **Evidence**: Supporting documents submitted by parties
  - Files, screenshots, communications
  - Submitted as JSON with IPFS URIs

```solidity
interface IEvidence {
    event MetaEvidence(uint256 indexed _metaEvidenceID, string _evidence);
    event Evidence(
        IArbitrator indexed _arbitrator, 
        uint256 indexed _evidenceGroupID, 
        address indexed _party, 
        string _evidence
    );
    event Dispute(
        IArbitrator indexed _arbitrator, 
        uint256 indexed _disputeID, 
        uint256 _metaEvidenceID, 
        uint256 _evidenceGroupID
    );
}
```

### 1.5 The extraData Parameter

The `extraData` parameter configures dispute creation:

| Bytes | Content | Description |
|-------|---------|-------------|
| 0-31 | Subcourt ID | Which specialized court handles the dispute |
| 32-63 | Juror Count | Minimum jurors required (usually 3) |

```javascript
// Generate extraData for a dispute
function generateExtraData(subcourtId, jurorCount) {
    return '0x' + 
        subcourtId.toString(16).padStart(64, '0') + 
        jurorCount.toString(16).padStart(64, '0');
}

// Example: General Court (0) with 3 jurors
const extraData = generateExtraData(0, 3);
// => 0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003
```

---

## 2. Contract Changes Needed

### 2.1 Current Escrow Architecture

Our current escrow (`~/kernle/kernle/commerce/escrow/`) has:

```
escrow/
├── __init__.py      # Module exports
├── abi.py           # Contract ABIs (placeholder)
├── events.py        # Event monitoring/parsing
└── service.py       # High-level escrow operations
```

**Current State Machine:**
```
Created → Funded → Accepted → Delivered → Released
    ↓         ↓         ↓          ↓
 (cancel)  Refunded  Disputed   Disputed
```

**Current Dispute Handling (simplified):**
```python
# From service.py
def resolve_dispute(
    self,
    escrow_address: str,
    recipient_address: str,     # Who gets the funds
    arbitrator_address: str,    # Centralized arbitrator
) -> TransactionResult:
    # ... direct transfer to recipient
```

### 2.2 New Contract: KernleEscrowArbitrable

We need a new Solidity contract that implements `IArbitrable`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@kleros/erc-792/contracts/IArbitrable.sol";
import "@kleros/erc-792/contracts/IArbitrator.sol";
import "@kleros/erc-792/contracts/erc-1497/IEvidence.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title KernleEscrowArbitrable
 * @notice Escrow contract with Kleros arbitration support
 * @dev Implements ERC-792 Arbitrable and ERC-1497 Evidence standards
 */
contract KernleEscrowArbitrable is IArbitrable, IEvidence, ReentrancyGuard {
    
    // =========================================================================
    // State Variables
    // =========================================================================
    
    enum Status {
        Created,
        Funded,
        Accepted,
        Delivered,
        WaitingClient,      // Client has raised dispute, waiting for worker fee
        WaitingWorker,      // Worker has raised dispute, waiting for client fee
        DisputeCreated,     // Dispute submitted to Kleros
        Resolved,           // Dispute resolved or released/refunded
        Released,           // Funds released to worker
        Refunded            // Funds refunded to client
    }
    
    enum RulingOptions {
        RefusedToArbitrate, // 0: Arbitrator refused (split funds)
        ClientWins,         // 1: Refund to client
        WorkerWins          // 2: Release to worker
    }
    
    bytes32 public immutable jobId;
    address public immutable client;
    address public worker;
    IERC20 public immutable usdc;
    IArbitrator public immutable arbitrator;
    
    uint256 public amount;
    uint256 public deadline;
    uint256 public deliveredAt;
    bytes32 public deliverableHash;
    
    Status public status;
    uint256 public disputeID;
    uint256 public metaEvidenceID;
    
    // Arbitration fee deposits
    uint256 public clientArbitrationFee;
    uint256 public workerArbitrationFee;
    
    // Timeouts
    uint256 public constant RECLAIM_PERIOD = 7 days;
    uint256 public constant FEE_DEPOSIT_PERIOD = 3 days;
    uint256 public constant APPROVAL_TIMEOUT = 7 days;
    
    uint256 public disputeRaisedAt;
    
    // Kleros court configuration
    bytes public arbitratorExtraData;
    uint256 constant NUMBER_OF_RULING_OPTIONS = 2;
    
    // =========================================================================
    // Events
    // =========================================================================
    
    event Funded(address indexed client, uint256 amount);
    event WorkerAssigned(address indexed worker);
    event Delivered(address indexed worker, bytes32 deliverableHash);
    event Released(address indexed worker, uint256 amount);
    event Refunded(address indexed client, uint256 amount);
    event DisputeRaised(address indexed disputant, Status previousStatus);
    event ArbitrationFeeDeposited(address indexed party, uint256 amount);
    event HasToPayFee(address indexed party);
    
    // =========================================================================
    // Errors
    // =========================================================================
    
    error Unauthorized();
    error InvalidStatus();
    error InvalidRuling();
    error DeadlineNotPassed();
    error TimeoutNotExpired();
    error InsufficientFee();
    error TransferFailed();
    error NotArbitrator();
    error FeeDepositPeriodExpired();
    error FeeDepositPeriodNotExpired();
    
    // =========================================================================
    // Modifiers
    // =========================================================================
    
    modifier onlyClient() {
        if (msg.sender != client) revert Unauthorized();
        _;
    }
    
    modifier onlyWorker() {
        if (msg.sender != worker) revert Unauthorized();
        _;
    }
    
    modifier onlyParty() {
        if (msg.sender != client && msg.sender != worker) revert Unauthorized();
        _;
    }
    
    modifier onlyArbitrator() {
        if (msg.sender != address(arbitrator)) revert NotArbitrator();
        _;
    }
    
    // =========================================================================
    // Constructor
    // =========================================================================
    
    constructor(
        bytes32 _jobId,
        address _client,
        IERC20 _usdc,
        IArbitrator _arbitrator,
        bytes memory _arbitratorExtraData,
        uint256 _amount,
        uint256 _deadline,
        string memory _metaEvidence
    ) {
        jobId = _jobId;
        client = _client;
        usdc = _usdc;
        arbitrator = _arbitrator;
        arbitratorExtraData = _arbitratorExtraData;
        amount = _amount;
        deadline = _deadline;
        status = Status.Created;
        
        // Emit meta evidence for this escrow
        emit MetaEvidence(0, _metaEvidence);
        metaEvidenceID = 0;
    }
    
    // =========================================================================
    // Core Escrow Functions
    // =========================================================================
    
    function fund() external onlyClient nonReentrant {
        if (status != Status.Created) revert InvalidStatus();
        
        bool success = usdc.transferFrom(client, address(this), amount);
        if (!success) revert TransferFailed();
        
        status = Status.Funded;
        emit Funded(client, amount);
    }
    
    function assignWorker(address _worker) external onlyClient {
        if (status != Status.Funded) revert InvalidStatus();
        if (_worker == address(0)) revert Unauthorized();
        
        worker = _worker;
        status = Status.Accepted;
        emit WorkerAssigned(_worker);
    }
    
    function deliver(bytes32 _deliverableHash) external onlyWorker {
        if (status != Status.Accepted) revert InvalidStatus();
        
        deliverableHash = _deliverableHash;
        deliveredAt = block.timestamp;
        status = Status.Delivered;
        emit Delivered(worker, _deliverableHash);
    }
    
    function release() external onlyClient nonReentrant {
        if (status != Status.Delivered) revert InvalidStatus();
        
        status = Status.Released;
        bool success = usdc.transfer(worker, amount);
        if (!success) revert TransferFailed();
        
        emit Released(worker, amount);
    }
    
    function autoRelease() external nonReentrant {
        if (status != Status.Delivered) revert InvalidStatus();
        if (block.timestamp < deliveredAt + APPROVAL_TIMEOUT) revert TimeoutNotExpired();
        
        status = Status.Released;
        bool success = usdc.transfer(worker, amount);
        if (!success) revert TransferFailed();
        
        emit Released(worker, amount);
    }
    
    function refund() external onlyClient nonReentrant {
        if (status != Status.Funded) revert InvalidStatus();
        
        status = Status.Refunded;
        bool success = usdc.transfer(client, amount);
        if (!success) revert TransferFailed();
        
        emit Refunded(client, amount);
    }
    
    // =========================================================================
    // Dispute Functions (ERC-792)
    // =========================================================================
    
    /**
     * @notice Raise a dispute by depositing arbitration fee
     * @dev Either party can initiate. Other party must deposit fee within FEE_DEPOSIT_PERIOD.
     */
    function raiseDispute() external payable onlyParty nonReentrant {
        // Can only dispute in Accepted or Delivered status
        if (status != Status.Accepted && status != Status.Delivered) revert InvalidStatus();
        
        uint256 arbitrationCost = arbitrator.arbitrationCost(arbitratorExtraData);
        if (msg.value < arbitrationCost) revert InsufficientFee();
        
        // Refund excess
        if (msg.value > arbitrationCost) {
            payable(msg.sender).transfer(msg.value - arbitrationCost);
        }
        
        Status previousStatus = status;
        disputeRaisedAt = block.timestamp;
        
        if (msg.sender == client) {
            clientArbitrationFee = arbitrationCost;
            status = Status.WaitingWorker;
            emit HasToPayFee(worker);
        } else {
            workerArbitrationFee = arbitrationCost;
            status = Status.WaitingClient;
            emit HasToPayFee(client);
        }
        
        emit DisputeRaised(msg.sender, previousStatus);
        emit ArbitrationFeeDeposited(msg.sender, arbitrationCost);
    }
    
    /**
     * @notice Deposit arbitration fee in response to other party's dispute
     */
    function depositArbitrationFee() external payable nonReentrant {
        if (status != Status.WaitingClient && status != Status.WaitingWorker) revert InvalidStatus();
        if (block.timestamp > disputeRaisedAt + FEE_DEPOSIT_PERIOD) revert FeeDepositPeriodExpired();
        
        uint256 arbitrationCost = arbitrator.arbitrationCost(arbitratorExtraData);
        if (msg.value < arbitrationCost) revert InsufficientFee();
        
        bool isClient = msg.sender == client;
        bool isWorker = msg.sender == worker;
        
        // Must be the party we're waiting for
        if (status == Status.WaitingClient && !isClient) revert Unauthorized();
        if (status == Status.WaitingWorker && !isWorker) revert Unauthorized();
        
        // Refund excess
        if (msg.value > arbitrationCost) {
            payable(msg.sender).transfer(msg.value - arbitrationCost);
        }
        
        if (isClient) {
            clientArbitrationFee = arbitrationCost;
        } else {
            workerArbitrationFee = arbitrationCost;
        }
        
        emit ArbitrationFeeDeposited(msg.sender, arbitrationCost);
        
        // Both parties have deposited - create dispute
        _createDispute();
    }
    
    /**
     * @notice Timeout if other party failed to deposit fee
     */
    function timeoutByClient() external onlyClient nonReentrant {
        if (status != Status.WaitingWorker) revert InvalidStatus();
        if (block.timestamp <= disputeRaisedAt + FEE_DEPOSIT_PERIOD) revert FeeDepositPeriodNotExpired();
        
        // Worker failed to deposit - client wins by default
        status = Status.Refunded;
        
        // Refund client's fee deposit
        payable(client).transfer(clientArbitrationFee);
        clientArbitrationFee = 0;
        
        // Refund escrowed USDC
        bool success = usdc.transfer(client, amount);
        if (!success) revert TransferFailed();
        
        emit Refunded(client, amount);
    }
    
    function timeoutByWorker() external onlyWorker nonReentrant {
        if (status != Status.WaitingClient) revert InvalidStatus();
        if (block.timestamp <= disputeRaisedAt + FEE_DEPOSIT_PERIOD) revert FeeDepositPeriodNotExpired();
        
        // Client failed to deposit - worker wins by default
        status = Status.Released;
        
        // Refund worker's fee deposit
        payable(worker).transfer(workerArbitrationFee);
        workerArbitrationFee = 0;
        
        // Release escrowed USDC
        bool success = usdc.transfer(worker, amount);
        if (!success) revert TransferFailed();
        
        emit Released(worker, amount);
    }
    
    /**
     * @dev Internal function to create dispute with Kleros
     */
    function _createDispute() internal {
        status = Status.DisputeCreated;
        
        // Create dispute with Kleros
        uint256 totalFees = clientArbitrationFee + workerArbitrationFee;
        disputeID = arbitrator.createDispute{value: totalFees}(
            NUMBER_OF_RULING_OPTIONS, 
            arbitratorExtraData
        );
        
        // Emit Dispute event (ERC-1497)
        emit Dispute(arbitrator, disputeID, metaEvidenceID, uint256(jobId));
    }
    
    /**
     * @notice Submit evidence for the dispute (ERC-1497)
     * @param _evidence IPFS URI to evidence JSON
     */
    function submitEvidence(string calldata _evidence) external onlyParty {
        if (status != Status.DisputeCreated) revert InvalidStatus();
        
        emit Evidence(arbitrator, uint256(jobId), msg.sender, _evidence);
    }
    
    /**
     * @notice Receive ruling from arbitrator (ERC-792)
     * @param _disputeID ID of the dispute
     * @param _ruling The ruling (0=refused, 1=client wins, 2=worker wins)
     */
    function rule(uint256 _disputeID, uint256 _ruling) external override onlyArbitrator {
        if (_disputeID != disputeID) revert InvalidRuling();
        if (status != Status.DisputeCreated) revert InvalidStatus();
        if (_ruling > NUMBER_OF_RULING_OPTIONS) revert InvalidRuling();
        
        status = Status.Resolved;
        
        if (_ruling == uint256(RulingOptions.ClientWins)) {
            // Refund to client
            bool success = usdc.transfer(client, amount);
            if (!success) revert TransferFailed();
            emit Refunded(client, amount);
        } else if (_ruling == uint256(RulingOptions.WorkerWins)) {
            // Release to worker
            bool success = usdc.transfer(worker, amount);
            if (!success) revert TransferFailed();
            emit Released(worker, amount);
        } else {
            // RefusedToArbitrate - split 50/50
            uint256 half = amount / 2;
            usdc.transfer(client, half);
            usdc.transfer(worker, amount - half);
        }
        
        emit Ruling(arbitrator, _disputeID, _ruling);
    }
    
    /**
     * @notice Appeal a ruling (optional)
     */
    function appeal() external payable onlyParty {
        if (status != Status.DisputeCreated) revert InvalidStatus();
        
        uint256 appealCost = arbitrator.appealCost(disputeID, arbitratorExtraData);
        if (msg.value < appealCost) revert InsufficientFee();
        
        arbitrator.appeal{value: msg.value}(disputeID, arbitratorExtraData);
    }
    
    // =========================================================================
    // View Functions
    // =========================================================================
    
    function getArbitrationCost() external view returns (uint256) {
        return arbitrator.arbitrationCost(arbitratorExtraData);
    }
    
    function getAppealCost() external view returns (uint256) {
        if (status != Status.DisputeCreated) return 0;
        return arbitrator.appealCost(disputeID, arbitratorExtraData);
    }
    
    function remainingTimeToDepositFee() external view returns (uint256) {
        if (status != Status.WaitingClient && status != Status.WaitingWorker) return 0;
        
        uint256 deadline_ = disputeRaisedAt + FEE_DEPOSIT_PERIOD;
        if (block.timestamp >= deadline_) return 0;
        return deadline_ - block.timestamp;
    }
    
    function remainingTimeToAutoRelease() external view returns (uint256) {
        if (status != Status.Delivered) return 0;
        
        uint256 autoReleaseTime = deliveredAt + APPROVAL_TIMEOUT;
        if (block.timestamp >= autoReleaseTime) return 0;
        return autoReleaseTime - block.timestamp;
    }
}
```

### 2.3 Updated Factory Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./KernleEscrowArbitrable.sol";
import "@kleros/erc-792/contracts/IArbitrator.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract KernleEscrowFactoryArbitrable is Ownable {
    IERC20 public immutable usdc;
    IArbitrator public arbitrator;
    bytes public defaultArbitratorExtraData;
    
    mapping(bytes32 => address) public escrows;
    address[] public allEscrows;
    
    event EscrowCreated(
        bytes32 indexed jobId,
        address indexed escrow,
        address indexed client,
        uint256 amount
    );
    event ArbitratorUpdated(address indexed oldArbitrator, address indexed newArbitrator);
    
    constructor(
        IERC20 _usdc,
        IArbitrator _arbitrator,
        bytes memory _defaultArbitratorExtraData
    ) Ownable(msg.sender) {
        usdc = _usdc;
        arbitrator = _arbitrator;
        defaultArbitratorExtraData = _defaultArbitratorExtraData;
    }
    
    function createEscrow(
        bytes32 jobId,
        uint256 amount,
        uint256 deadline,
        string calldata metaEvidence
    ) external returns (address) {
        require(escrows[jobId] == address(0), "Escrow exists");
        
        KernleEscrowArbitrable escrow = new KernleEscrowArbitrable(
            jobId,
            msg.sender,
            usdc,
            arbitrator,
            defaultArbitratorExtraData,
            amount,
            deadline,
            metaEvidence
        );
        
        address escrowAddr = address(escrow);
        escrows[jobId] = escrowAddr;
        allEscrows.push(escrowAddr);
        
        emit EscrowCreated(jobId, escrowAddr, msg.sender, amount);
        return escrowAddr;
    }
    
    function setArbitrator(IArbitrator _arbitrator) external onlyOwner {
        emit ArbitratorUpdated(address(arbitrator), address(_arbitrator));
        arbitrator = _arbitrator;
    }
    
    function setDefaultArbitratorExtraData(bytes calldata _data) external onlyOwner {
        defaultArbitratorExtraData = _data;
    }
    
    function getEscrow(bytes32 jobId) external view returns (address) {
        return escrows[jobId];
    }
    
    function escrowCount() external view returns (uint256) {
        return allEscrows.length;
    }
}
```

### 2.4 Python Service Updates

Update `~/kernle/kernle/commerce/escrow/service.py`:

```python
# New imports and additions to EscrowService

# === New Methods for Kleros Integration ===

def get_arbitration_cost(self, escrow_address: str) -> Decimal:
    """Get current arbitration cost in ETH."""
    # TODO: Implement with Web3
    # escrow = self._get_escrow(escrow_address)
    # cost_wei = escrow.functions.getArbitrationCost().call()
    # return Decimal(cost_wei) / Decimal(10**18)
    return Decimal("0.1")  # Placeholder ~0.1 ETH

def raise_dispute(
    self,
    escrow_address: str,
    disputant_address: str,
    arbitration_fee_eth: Decimal,
) -> TransactionResult:
    """Raise a dispute and deposit arbitration fee.
    
    Args:
        escrow_address: Escrow contract address
        disputant_address: Address raising the dispute
        arbitration_fee_eth: Arbitration fee in ETH
    """
    logger.info(f"Raising dispute on escrow {escrow_address}")
    
    # TODO: Implement with Web3
    # escrow = self._get_escrow(escrow_address)
    # tx = escrow.functions.raiseDispute().build_transaction({
    #     "from": disputant_address,
    #     "value": int(arbitration_fee_eth * 10**18),
    #     "gas": 200000,
    # })
    
    return TransactionResult(success=True, tx_hash=f"0x{'0' * 64}")

def deposit_arbitration_fee(
    self,
    escrow_address: str,
    party_address: str,
    arbitration_fee_eth: Decimal,
) -> TransactionResult:
    """Deposit arbitration fee in response to dispute."""
    logger.info(f"Depositing arbitration fee for {party_address}")
    
    # TODO: Implement
    return TransactionResult(success=True, tx_hash=f"0x{'0' * 64}")

def submit_evidence(
    self,
    escrow_address: str,
    party_address: str,
    evidence_uri: str,
) -> TransactionResult:
    """Submit evidence for a dispute.
    
    Args:
        escrow_address: Escrow contract address
        party_address: Party submitting evidence
        evidence_uri: IPFS URI to evidence JSON
    """
    logger.info(f"Submitting evidence to escrow {escrow_address}")
    
    # TODO: Implement
    return TransactionResult(success=True, tx_hash=f"0x{'0' * 64}")

def appeal_ruling(
    self,
    escrow_address: str,
    party_address: str,
    appeal_fee_eth: Decimal,
) -> TransactionResult:
    """Appeal a ruling."""
    logger.info(f"Appealing ruling on escrow {escrow_address}")
    
    # TODO: Implement
    return TransactionResult(success=True, tx_hash=f"0x{'0' * 64}")

def timeout_dispute(
    self,
    escrow_address: str,
    caller_address: str,
) -> TransactionResult:
    """Claim timeout if other party failed to deposit fee."""
    logger.info(f"Claiming timeout on escrow {escrow_address}")
    
    # TODO: Implement - call timeoutByClient or timeoutByWorker
    return TransactionResult(success=True, tx_hash=f"0x{'0' * 64}")
```

### 2.5 New ABI Additions

Add to `~/kernle/kernle/commerce/escrow/abi.py`:

```python
# Kleros IArbitrator ABI (subset needed for interactions)
KLEROS_ARBITRATOR_ABI: ABI = [
    {
        "type": "function",
        "name": "arbitrationCost",
        "inputs": [{"name": "_extraData", "type": "bytes"}],
        "outputs": [{"name": "cost", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "appealCost",
        "inputs": [
            {"name": "_disputeID", "type": "uint256"},
            {"name": "_extraData", "type": "bytes"},
        ],
        "outputs": [{"name": "cost", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "disputeStatus",
        "inputs": [{"name": "_disputeID", "type": "uint256"}],
        "outputs": [{"name": "status", "type": "uint8"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "currentRuling",
        "inputs": [{"name": "_disputeID", "type": "uint256"}],
        "outputs": [{"name": "ruling", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "appealPeriod",
        "inputs": [{"name": "_disputeID", "type": "uint256"}],
        "outputs": [
            {"name": "start", "type": "uint256"},
            {"name": "end", "type": "uint256"},
        ],
        "stateMutability": "view",
    },
]

# ERC-792 Arbitrable events
ARBITRABLE_EVENTS: ABI = [
    {
        "type": "event",
        "name": "Ruling",
        "inputs": [
            {"name": "_arbitrator", "type": "address", "indexed": True},
            {"name": "_disputeID", "type": "uint256", "indexed": True},
            {"name": "_ruling", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Dispute",
        "inputs": [
            {"name": "_arbitrator", "type": "address", "indexed": True},
            {"name": "_disputeID", "type": "uint256", "indexed": True},
            {"name": "_metaEvidenceID", "type": "uint256", "indexed": False},
            {"name": "_evidenceGroupID", "type": "uint256", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "Evidence",
        "inputs": [
            {"name": "_arbitrator", "type": "address", "indexed": True},
            {"name": "_evidenceGroupID", "type": "uint256", "indexed": True},
            {"name": "_party", "type": "address", "indexed": True},
            {"name": "_evidence", "type": "string", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "MetaEvidence",
        "inputs": [
            {"name": "_metaEvidenceID", "type": "uint256", "indexed": True},
            {"name": "_evidence", "type": "string", "indexed": False},
        ],
    },
]
```

---

## 3. Dispute Flow

### 3.1 High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HAPPY PATH (No Dispute)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Client creates job → Funds escrow → Assigns worker → Worker delivers      │
│                                            ↓                                 │
│                                  Client approves → Funds released           │
│                                       OR                                     │
│                                  7 days pass → Auto-release                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DISPUTE PATH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INITIATION                                                               │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │ Either party raises dispute + deposits arbitration fee (~0.1 ETH) │    │
│     └───────────────────────────────┬──────────────────────────────────┘    │
│                                     ↓                                        │
│  2. COUNTER-DEPOSIT (3 days)                                                │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │ Other party deposits fee    │  OR  │  Fails to deposit           │    │
│     │         ↓                   │      │         ↓                   │    │
│     │   Dispute created           │      │   Initiator wins by default │    │
│     └───────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                        │
│  3. EVIDENCE SUBMISSION                                                      │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │ Both parties submit evidence to IPFS, emit Evidence events       │    │
│     │ • Original contract/agreement                                     │    │
│     │ • Communications, screenshots                                     │    │
│     │ • Deliverable review                                              │    │
│     └───────────────────────────────┬──────────────────────────────────┘    │
│                                     ↓                                        │
│  4. KLEROS ARBITRATION                                                       │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │ Kleros jurors review evidence and vote                           │    │
│     │ • Jury drawn from PNK stakers                                    │    │
│     │ • Majority vote determines ruling                                │    │
│     │ • Appeal period after ruling                                     │    │
│     └───────────────────────────────┬──────────────────────────────────┘    │
│                                     ↓                                        │
│  5. RULING ENFORCEMENT                                                       │
│     ┌──────────────────────────────────────────────────────────────────┐    │
│     │ Arbitrator calls rule() on escrow contract                       │    │
│     │                                                                   │    │
│     │   Ruling 0: Refused → Split 50/50                                │    │
│     │   Ruling 1: Client wins → Refund to client                       │    │
│     │   Ruling 2: Worker wins → Release to worker                      │    │
│     └──────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Detailed State Machine

```
                                    ┌──────────┐
                                    │ Created  │
                                    └────┬─────┘
                                         │ fund()
                                         ↓
                                    ┌──────────┐
                           refund() │  Funded  │
                   ┌────────────────┴────┬─────┘
                   ↓                     │ assignWorker()
             ┌──────────┐                ↓
             │ Refunded │           ┌──────────┐
             └──────────┘           │ Accepted │
                                    └────┬─────┘
                                         │ deliver()
                                         ↓
                                    ┌───────────┐
                                    │ Delivered │
                                    └─────┬─────┘
                            ┌─────────────┼─────────────┐
                            ↓             ↓             ↓
                       release()    autoRelease()  raiseDispute()
                            ↓             ↓             ↓
                       ┌──────────┐  ┌──────────┐  ┌─────────────────┐
                       │ Released │  │ Released │  │ WaitingClient/  │
                       └──────────┘  └──────────┘  │ WaitingWorker   │
                                                   └────────┬────────┘
                                          ┌─────────────────┼─────────────────┐
                                          ↓                 ↓                 ↓
                                   depositFee()        timeout()         (no action)
                                          ↓                 ↓                 ↓
                                   ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
                                   │ Dispute      │  │ Released/    │  │ (waits for  │
                                   │ Created      │  │ Refunded     │  │  timeout)   │
                                   └──────┬───────┘  └──────────────┘  └─────────────┘
                                          │
                                          │ rule() from Kleros
                                          ↓
                                   ┌──────────────┐
                                   │   Resolved   │ → Funds transferred based on ruling
                                   └──────────────┘
```

### 3.3 Evidence Submission Guide

#### MetaEvidence JSON (Created at escrow deployment)

```json
{
  "category": "Freelance Work",
  "title": "Kernle Job Dispute",
  "description": "A dispute has arisen regarding the completion of a job on Kernle Commerce. The client claims the deliverable does not meet the agreed specifications. The worker claims the work was completed as requested.",
  "question": "Did the worker complete the job according to the agreed specifications?",
  "rulingOptions": {
    "type": "single-select",
    "titles": ["Refund Client", "Pay Worker"],
    "descriptions": [
      "The work was not completed satisfactorily. Refund the full escrow amount to the client.",
      "The work was completed as specified. Release the escrow payment to the worker."
    ]
  },
  "fileURI": "/ipfs/Qm.../job-spec.pdf",
  "evidenceDisplayInterfaceURI": "https://app.kernle.ai/dispute-viewer",
  "dynamicScriptURI": "/ipfs/Qm.../dynamic-meta.js"
}
```

#### Evidence JSON (Submitted during dispute)

```json
{
  "name": "Deliverable Review",
  "description": "Analysis of submitted work vs requirements",
  "fileURI": "/ipfs/Qm.../evidence.pdf",
  "fileTypeExtension": "pdf",
  "fileHash": "Qm..."
}
```

### 3.4 Timeline Summary

| Phase | Duration | Description |
|-------|----------|-------------|
| Fee Deposit | 3 days | Other party must deposit arbitration fee |
| Evidence | ~7 days | Parties submit evidence (overlaps with voting) |
| Voting | ~7 days | Jurors review and vote |
| Appeal | ~3-7 days | Losing party can appeal |
| **Total** | **~14-21 days** | End-to-end dispute resolution |

---

## 4. Cost Considerations

### 4.1 Arbitration Fee Structure

Kleros arbitration costs depend on:

1. **Subcourt selection** - Specialized courts may cost more
2. **Number of jurors** - More jurors = higher cost (but more reliable)
3. **ETH price** - Fees are paid in ETH

**Estimated Costs (as of 2024-2025):**

| Court | Jurors | Estimated Cost |
|-------|--------|----------------|
| General Court | 3 | ~0.05-0.15 ETH |
| Technical | 3 | ~0.08-0.20 ETH |
| Curation | 3 | ~0.03-0.10 ETH |

### 4.2 Who Pays?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEE PAYMENT MODEL                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BOTH PARTIES DEPOSIT:                                                       │
│  • Initiating party deposits fee first                                       │
│  • Responding party deposits within 3 days                                   │
│  • Total: 2x arbitration cost collected                                      │
│                                                                              │
│  AFTER RULING:                                                               │
│  • Winner gets their fee refunded from loser's deposit                       │
│  • Net cost to loser: 1x arbitration fee + escrow amount                     │
│  • Net cost to winner: 0 (refunded)                                         │
│                                                                              │
│  IF ONE PARTY DOESN'T DEPOSIT:                                              │
│  • Party who deposited wins by default                                       │
│  • Gets full refund of fee                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Gas Costs (Base Network)

| Operation | Estimated Gas | @ 0.01 gwei | @ 0.1 gwei |
|-----------|---------------|-------------|------------|
| Create Escrow | 250,000 | ~$0.01 | ~$0.10 |
| Fund | 100,000 | <$0.01 | ~$0.04 |
| Raise Dispute | 200,000 | ~$0.01 | ~$0.08 |
| Submit Evidence | 50,000 | <$0.01 | ~$0.02 |
| Rule (Kleros) | 150,000 | ~$0.01 | ~$0.06 |

**Note:** Base network has very low gas costs compared to Ethereum mainnet.

### 4.4 Economic Recommendations

1. **Minimum dispute value**: Only disputes worth >$100 make economic sense given ~$50-200 arbitration costs

2. **Fee escrow**: Consider requiring parties to pre-deposit arbitration fees upfront to ensure dispute resolution is always possible

3. **Platform fee**: Kernle could charge a small platform fee (1-3%) to subsidize dispute resolution for smaller amounts

4. **Incentive alignment**: The "loser pays" model discourages frivolous disputes

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Create Solidity contracts (`KernleEscrowArbitrable.sol`, factory)
- [ ] Deploy Centralized Arbitrator for testing
- [ ] Write unit tests with Foundry/Hardhat
- [ ] Deploy to Base Sepolia testnet

### Phase 2: Python Integration (Week 3)

- [ ] Update `EscrowService` with new methods
- [ ] Add event monitoring for Kleros events
- [ ] Create evidence submission helpers (IPFS upload)
- [ ] Add CLI commands for dispute management

### Phase 3: UI/UX (Week 4)

- [ ] Build dispute status dashboard
- [ ] Create evidence submission flow
- [ ] Display arbitration costs to users
- [ ] Add timeline/countdown displays

### Phase 4: Testing & Audit (Week 5-6)

- [ ] Integration tests with Kleros testnet
- [ ] Security review
- [ ] Gas optimization
- [ ] Documentation

### Phase 5: Mainnet (Week 7+)

- [ ] Deploy to Base mainnet
- [ ] Configure Kleros Court connection
- [ ] Monitor first disputes
- [ ] Iterate based on feedback

---

## 6. Security Considerations

### 6.1 Smart Contract Security

| Risk | Mitigation |
|------|------------|
| Reentrancy | Use OpenZeppelin ReentrancyGuard on all state-changing functions |
| Front-running | N/A - disputes are time-locked |
| Integer overflow | Solidity 0.8.x has built-in overflow checks |
| Access control | Strict modifiers (onlyClient, onlyWorker, onlyArbitrator) |
| Evidence tampering | Content-addressed storage (IPFS) with hashes on-chain |

### 6.2 Arbitration Security

| Risk | Mitigation |
|------|------------|
| Juror collusion | Kleros uses random juror selection + token staking |
| Bribery | Jurors lose stake if vote incoherently |
| Arbitrator manipulation | Kleros is decentralized; no single point of control |
| Appeal abuse | Appeals cost more each round, discouraging abuse |

### 6.3 Operational Security

- **Private key management**: Use hardware wallets for factory owner keys
- **Upgrade path**: Factory owner can update arbitrator address (for migration to new Kleros versions)
- **Emergency pause**: Consider adding pausable functionality for critical bugs

---

## Appendix

### A.1 Kleros Court Deployments

**Ethereum Mainnet:**
- Kleros Arbitrator: `0x988b3A538b618C7A603e1c11Ab82Cd16dbE28069`

**Arbitrum One (Kleros 2.0):**
- Deployed November 2024 (check Kleros docs for address)

**Gnosis Chain:**
- Alternative for lower costs

**Testnets:**
- Use Centralized Arbitrator for testing

### A.2 Subcourt IDs (Ethereum Mainnet)

| ID | Court Name |
|----|------------|
| 0 | General Court |
| 1 | Blockchain |
| 2 | Non-Technical |
| 3 | Token Listing |
| 4 | Technical |
| 5 | Marketing Services |
| 6 | English Language |
| ... | (see klerosboard.com) |

### A.3 Related Standards

- [EIP-792: Arbitration Standard](https://github.com/ethereum/EIPs/issues/792)
- [EIP-1497: Evidence Standard](https://github.com/ethereum/EIPs/issues/1497)
- [Kleros Documentation](https://docs.kleros.io)
- [erc-792 Reference Implementation](https://github.com/kleros/erc-792)

### A.4 Example MetaEvidence for Kernle Jobs

```json
{
  "category": "Freelance/Gig Work",
  "title": "Kernle AI Agent Job Dispute",
  "description": "This dispute involves a job posted on Kernle Commerce, a marketplace for AI agent services. A client has hired an AI agent (worker) to complete a specified task with defined deliverables.",
  "question": "Based on the job specification and submitted deliverable, has the worker satisfactorily completed the contracted work?",
  "rulingOptions": {
    "type": "single-select",
    "titles": [
      "Refund to Client",
      "Release to Worker"
    ],
    "descriptions": [
      "The submitted work does not meet the specifications outlined in the job contract. The escrow should be refunded to the client.",
      "The submitted work satisfies the job requirements. The escrow should be released to the worker."
    ]
  },
  "fileURI": "/ipfs/QmJobSpecification...",
  "evidenceDisplayInterfaceURI": "https://app.kernle.ai/evidence-viewer?job={jobId}",
  "aliases": {
    "client": "Client (Job Poster)",
    "worker": "Worker (AI Agent)"
  },
  "specification": {
    "jobId": "{jobId}",
    "platform": "Kernle Commerce",
    "network": "Base",
    "escrowContract": "{escrowAddress}"
  }
}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-31 | Initial draft |
