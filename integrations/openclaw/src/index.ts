import type { PluginApi, PluginConfig, ToolResultInfo, ToolResultPersistResult } from "./types.js";
import { createBeforeAgentStart } from "./hooks/before-agent-start.js";
import { createAgentEnd } from "./hooks/agent-end.js";
import { createBeforeToolCall } from "./hooks/before-tool-call.js";
import { resolveStackId } from "./stack-id.js";

const TRANSCRIPT_TRIM_LIMIT = 2000;

/**
 * Kernle OpenClaw plugin entry point.
 *
 * Registers all hooks:
 * - before_agent_start: Load Kernle memory into session context
 * - agent_end: Auto-checkpoint on session end
 * - before_tool_call: Block native memory writes, capture into Kernle
 * - tool_result_persist: Trim large kernle output in transcript
 */
export function register(api: PluginApi): void {
  const config: PluginConfig = api.getConfig();

  // 1. Memory loading at session start
  api.on("before_agent_start", createBeforeAgentStart(config));

  // 2. Auto-checkpoint on session end
  api.on("agent_end", createAgentEnd(config));

  // 3. Block native memory writes
  //    Resolve stackId eagerly for the tool call hook (no session context available)
  const stackId = resolveStackId(config, {});
  api.on("before_tool_call", createBeforeToolCall(config, stackId));

  // 4. Trim large kernle output in transcript
  api.on("tool_result_persist", (info: ToolResultInfo): ToolResultPersistResult | void => {
    // Only trim kernle-related output
    if (!info.toolName.startsWith("kernle") && !info.result.includes("# Kernle Memory")) {
      return;
    }

    if (info.result.length > TRANSCRIPT_TRIM_LIMIT) {
      return {
        result:
          info.result.slice(0, TRANSCRIPT_TRIM_LIMIT) +
          `\n\n[Trimmed: ${info.result.length} chars → ${TRANSCRIPT_TRIM_LIMIT}]`,
      };
    }
  });
}
