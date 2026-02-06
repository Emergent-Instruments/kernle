"""kernle.stack - Memory stack implementations.

The default SQLiteStack wraps SQLiteStorage and conforms to StackProtocol.
"""

from kernle.stack.sqlite_stack import SQLiteStack

__all__ = ["SQLiteStack"]
