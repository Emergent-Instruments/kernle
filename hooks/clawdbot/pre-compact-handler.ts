/**
 * kernle-checkpoint hook: Automatically save checkpoint before context compaction
 *
 * This hook fires before compaction to ensure the agent's working state is saved.
 * Combined with kernle-load hook, this enables seamless context transitions.
 *
 * Installation: Copy alongside handler.ts in the kernle-load hook directory
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { HookHandler } from "../../hooks.js";

const execAsync = promisify(exec);

interface PreCompactContext {
  workspaceDir?: string;
  sessionKey?: string;
  sessionId?: string;
  // Context summary from the compaction - what was being discussed
  contextSummary?: string;
  // Recent messages or topics
  recentTopics?: string[];
}

/**
 * Extract agent ID from session key (e.g., "agent:claire:main" -> "claire")
 */
function extractAgentId(sessionKey: string | undefined, workspaceDir: string | undefined): string {
  if (sessionKey) {
    const parts = sessionKey.split(":");
    if (parts.length >= 2 && parts[0] === "agent") {
      return parts[1];
    }
  }

  if (workspaceDir) {
    const dirName = workspaceDir.split("/").filter(Boolean).pop();
    if (dirName && dirName !== "clawd" && dirName !== "workspace") {
      return dirName;
    }
  }

  return "main";
}

/**
 * Save checkpoint before compaction
 */
async function saveCheckpoint(
  agentId: string,
  task: string,
  context?: string
): Promise<boolean> {
  try {
    // Build command with proper escaping
    const escapedTask = task.replace(/'/g, "'\\''");
    const contextArg = context
      ? ` --context '${context.replace(/'/g, "'\\''")}'`
      : "";

    const cmd = `kernle -a ${agentId} checkpoint '${escapedTask}'${contextArg}`;

    await execAsync(cmd, {
      timeout: 5000,
      maxBuffer: 1024 * 64,
    });

    return true;
  } catch (error: any) {
    const stderr = error.stderr || error.message || "";
    if (!stderr.includes("command not found")) {
      console.warn(`[kernle-checkpoint] Failed to save checkpoint for '${agentId}':`, stderr);
    }
    return false;
  }
}

/**
 * Hook handler: save checkpoint before context compaction
 */
const kernlePreCompactHook: HookHandler = async (event) => {
  // Handle pre-compaction events
  // Note: Event type may vary by Moltbot version
  if (
    !(
      (event.type === "context" && event.action === "pre-compact") ||
      (event.type === "agent" && event.action === "pre-compact") ||
      (event.type === "compaction" && event.action === "before")
    )
  ) {
    return;
  }

  const context = event.context as PreCompactContext;
  const { workspaceDir, sessionKey, contextSummary, recentTopics } = context;
  const agentId = extractAgentId(sessionKey, workspaceDir);

  // Build task description from available context
  let task = "Context compaction - working state preserved";

  if (contextSummary) {
    task = contextSummary.slice(0, 200);
  } else if (recentTopics && recentTopics.length > 0) {
    task = `Working on: ${recentTopics.slice(0, 3).join(", ")}`;
  }

  // Include recent topics as context if available
  const taskContext = recentTopics
    ? `Recent work: ${recentTopics.join("; ")}`
    : undefined;

  // Save checkpoint
  const saved = await saveCheckpoint(agentId, task, taskContext);

  if (saved) {
    console.log(`[kernle-checkpoint] Saved pre-compaction checkpoint for '${agentId}'`);
  }
};

export default kernlePreCompactHook;
