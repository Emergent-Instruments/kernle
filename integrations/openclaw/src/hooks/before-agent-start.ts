import type { BeforeAgentStartContext, BeforeAgentStartResult, PluginConfig } from "../types.js";
import { KernleBridge } from "../services/kernle-bridge.js";
import { resolveStackId } from "../stack-id.js";

/**
 * before_agent_start hook: Load Kernle memory and inject it as prepended context.
 *
 * Runs `kernle -s {stackId} load --budget {budget}` and returns the output
 * as `prependContext` so it appears in the agent's system prompt.
 *
 * Graceful degradation: if kernle is not installed, the stack doesn't exist,
 * or the command times out, the session continues without memory.
 */
export function createBeforeAgentStart(config: PluginConfig) {
  const bridge = new KernleBridge({
    kernleBin: config.kernleBin,
    timeout: config.timeout,
  });

  return async (
    context: BeforeAgentStartContext
  ): Promise<BeforeAgentStartResult | void> => {
    const stackId = resolveStackId(config, context);
    const budget = config.tokenBudget ?? 8000;

    const memory = await bridge.load(stackId, budget);

    if (!memory) {
      return;
    }

    return { prependContext: memory };
  };
}
