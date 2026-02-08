import type { ToolCallInfo, BeforeToolCallResult, PluginConfig } from "../types.js";
import { KernleBridge } from "../services/kernle-bridge.js";

/** File paths that are part of OpenClaw's native memory system. */
const MEMORY_PATTERNS = [
  /^memory\//,
  /\/memory\//,
  /MEMORY\.md$/,
];

/**
 * before_tool_call hook: Intercept native memory file writes.
 *
 * When the agent tries to write to `memory/` or `MEMORY.md`, this hook
 * captures the content as a Kernle raw entry and blocks the native write.
 *
 * This prevents the agent from maintaining two parallel memory systems.
 * All memory goes through Kernle's structured pipeline instead.
 *
 * Note: This hook depends on `before_tool_call` being wired in OpenClaw
 * (partially wired as of PR #6570). If it doesn't fire, loading and
 * checkpointing still work via the other hooks.
 */
export function createBeforeToolCall(config: PluginConfig, stackId: string) {
  const bridge = new KernleBridge({
    kernleBin: config.kernleBin,
    timeout: config.timeout,
  });

  return async (tool: ToolCallInfo): Promise<BeforeToolCallResult | void> => {
    // Only intercept file write operations
    const writeTools = ["write_file", "edit_file", "create_file"];
    if (!writeTools.includes(tool.name)) {
      return;
    }

    // Check if the target path is a memory file
    const filePath = (tool.arguments.file_path ?? tool.arguments.path ?? "") as string;
    if (!isMemoryPath(filePath)) {
      return;
    }

    // Capture content into Kernle as a raw entry
    const content = (tool.arguments.content ?? tool.arguments.new_string ?? "") as string;
    if (content) {
      const truncated = content.length > 2000
        ? content.slice(0, 2000) + "\n[truncated]"
        : content;
      await bridge.raw(stackId, `[memory-capture] ${filePath}\n\n${truncated}`);
    }

    return {
      blocked: true,
      message:
        "Memory writes are handled by Kernle. Use `kernle raw`, `kernle episode`, " +
        "or `kernle note` to record memories. Loading and checkpointing are automatic.",
    };
  };
}

function isMemoryPath(filePath: string): boolean {
  return MEMORY_PATTERNS.some((pattern) => pattern.test(filePath));
}
