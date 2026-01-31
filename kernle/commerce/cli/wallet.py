"""
Wallet CLI commands for Kernle Commerce.

Provides command-line interface for wallet operations:
- kernle wallet balance
- kernle wallet address
- kernle wallet status
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from kernle import Kernle


logger = logging.getLogger(__name__)


def _get_wallet_service():
    """Get a wallet service instance.
    
    For now, returns an InMemory storage-backed service.
    In production, this will use Supabase storage.
    """
    from kernle.commerce.wallet.service import WalletService
    from kernle.commerce.wallet.storage import InMemoryWalletStorage
    
    storage = InMemoryWalletStorage()
    return WalletService(storage)


def cmd_wallet(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle wallet subcommands."""
    action = args.wallet_action
    
    if action == "balance":
        _wallet_balance(args, k)
    elif action == "address":
        _wallet_address(args, k)
    elif action == "status":
        _wallet_status(args, k)
    else:
        print(f"Unknown wallet action: {action}")
        print("Available actions: balance, address, status")


def _wallet_balance(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show USDC balance for the agent's wallet."""
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)
    
    try:
        service = _get_wallet_service()
        balance = service.get_balance_for_agent(agent_id)
        
        if output_json:
            result = {
                "agent_id": agent_id,
                "wallet_address": balance.wallet_address,
                "usdc_balance": str(balance.usdc_balance),
                "eth_balance": str(balance.eth_balance),
                "chain": balance.chain,
                "as_of": balance.as_of.isoformat() if balance.as_of else None,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"💰 Wallet Balance for {agent_id}")
            print("=" * 40)
            print(f"  USDC:    ${balance.usdc_balance:.2f}")
            print(f"  ETH:     {balance.eth_balance:.6f}")
            print(f"  Chain:   {balance.chain}")
            print(f"  Address: {balance.wallet_address}")
    
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            if "not found" in str(e).lower():
                print(f"⚠️  No wallet found for agent '{agent_id}'")
                print("")
                print("A wallet is created when you register with Kernle Commerce.")
                print("See: https://docs.kernle.ai/commerce/getting-started")
            else:
                print(f"❌ Error: {e}")


def _wallet_address(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show wallet address for the agent."""
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)
    
    try:
        service = _get_wallet_service()
        wallet = service.get_wallet_for_agent(agent_id)
        
        if output_json:
            result = {
                "agent_id": agent_id,
                "wallet_address": wallet.wallet_address,
                "chain": wallet.chain,
            }
            print(json.dumps(result, indent=2))
        else:
            print(wallet.wallet_address)
    
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            if "not found" in str(e).lower():
                print(f"⚠️  No wallet found for agent '{agent_id}'")
            else:
                print(f"❌ Error: {e}")


def _wallet_status(args: "argparse.Namespace", k: "Kernle") -> None:
    """Show full wallet status including limits."""
    agent_id = k.agent_id
    output_json = getattr(args, "json", False)
    
    try:
        service = _get_wallet_service()
        wallet = service.get_wallet_for_agent(agent_id)
        
        if output_json:
            result = wallet.to_dict()
            result["is_active"] = wallet.is_active
            result["is_claimed"] = wallet.is_claimed
            result["can_transact"] = wallet.can_transact
            print(json.dumps(result, indent=2, default=str))
        else:
            status_emoji = {
                "active": "🟢",
                "pending_claim": "🟡",
                "paused": "🟠",
                "frozen": "🔴",
            }.get(wallet.status, "⚪")
            
            print(f"💳 Wallet Status for {agent_id}")
            print("=" * 50)
            print(f"  Address:         {wallet.wallet_address}")
            print(f"  Chain:           {wallet.chain}")
            print(f"  Status:          {status_emoji} {wallet.status}")
            print(f"  Can Transact:    {'Yes' if wallet.can_transact else 'No'}")
            print("")
            print("Spending Limits:")
            print(f"  Per Transaction: ${wallet.spending_limit_per_tx:.2f} USDC")
            print(f"  Daily:           ${wallet.spending_limit_daily:.2f} USDC")
            
            if wallet.owner_eoa:
                print("")
                print(f"Owner (EOA):       {wallet.owner_eoa}")
            
            if wallet.created_at:
                print("")
                print(f"Created:           {wallet.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
            
            if wallet.claimed_at:
                print(f"Claimed:           {wallet.claimed_at.strftime('%Y-%m-%d %H:%M UTC')}")
    
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            if "not found" in str(e).lower():
                print(f"⚠️  No wallet found for agent '{agent_id}'")
                print("")
                print("A wallet is created when you register with Kernle Commerce.")
                print("See: https://docs.kernle.ai/commerce/getting-started")
            else:
                print(f"❌ Error: {e}")
