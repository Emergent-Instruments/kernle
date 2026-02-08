import type { PluginConfig, BeforeAgentStartContext } from "./types.js";

/**
 * Resolve the Kernle stack ID from available context.
 *
 * Resolution order:
 * 1. Explicit config (`stackId` in plugin config)
 * 2. `KERNLE_STACK_ID` environment variable
 * 3. Agent ID from OpenClaw session key (e.g., "agent:ash:main" -> "ash")
 * 4. Workspace directory name
 * 5. Fallback: "main"
 */
export function resolveStackId(
  config: PluginConfig,
  context: Pick<BeforeAgentStartContext, "sessionKey" | "workspaceDir">
): string {
  // 1. Explicit config
  if (config.stackId) {
    return config.stackId;
  }

  // 2. Environment variable
  const envId = process.env.KERNLE_STACK_ID;
  if (envId) {
    return envId;
  }

  // 3. Session key (e.g., "agent:ash:main" -> "ash")
  if (context.sessionKey) {
    const parts = context.sessionKey.split(":");
    if (parts.length >= 2 && parts[0] === "agent") {
      return parts[1];
    }
  }

  // 4. Workspace directory name
  if (context.workspaceDir) {
    const dirName = context.workspaceDir.split("/").filter(Boolean).pop();
    if (dirName && dirName !== "workspace") {
      return dirName;
    }
  }

  // 5. Fallback
  return "main";
}
