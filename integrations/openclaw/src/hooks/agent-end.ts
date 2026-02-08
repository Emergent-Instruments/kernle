import type { AgentEndContext, PluginConfig } from "../types.js";
import { KernleBridge } from "../services/kernle-bridge.js";
import { resolveStackId } from "../stack-id.js";

/**
 * agent_end hook: Auto-checkpoint and raw capture on session end.
 *
 * Extracts a summary from the conversation (last user message as task,
 * last assistant message for context) and saves a checkpoint.
 * Also saves a raw entry marking session completion.
 *
 * Non-blocking — failures are silently ignored.
 */
export function createAgentEnd(config: PluginConfig) {
  const bridge = new KernleBridge({
    kernleBin: config.kernleBin,
    timeout: config.timeout,
  });

  return async (context: AgentEndContext): Promise<void> => {
    const stackId = resolveStackId(config, context);
    const messages = context.messages ?? [];

    // Extract summary from conversation
    const lastUserMsg = findLast(messages, "user");
    const lastAssistantMsg = findLast(messages, "assistant");

    const task = lastUserMsg
      ? truncate(lastUserMsg.content, 200)
      : "Session ended";

    const assistantContext = lastAssistantMsg
      ? truncate(lastAssistantMsg.content, 500)
      : undefined;

    // Save checkpoint and raw entry in parallel
    await Promise.allSettled([
      bridge.checkpoint(stackId, task, assistantContext),
      bridge.raw(stackId, `Session ended. Task: ${task}`),
    ]);
  };
}

function findLast(
  messages: { role: string; content: string }[],
  role: string
): { role: string; content: string } | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === role) {
      return messages[i];
    }
  }
  return undefined;
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}
