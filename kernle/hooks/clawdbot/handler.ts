/**
 * kernle-load hook: Automatically inject Kernle memory into session context
 */

import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { HookHandler } from "../../hooks.js";

const execAsync = promisify(exec);

interface BootstrapContext {
  workspaceDir?: string;
  sessionKey?: string;
  sessionId?: string;
  bootstrapFiles?: Array<{ path: string; content: string; virtual?: boolean }>;
  config?: any;
}

/**
 * Extract stack ID from session key (e.g., "agent:ash:main" -> "ash")
 */
function extractStackId(sessionKey: string | undefined, workspaceDir: string | undefined): string {
  if (sessionKey) {
    const parts = sessionKey.split(":");
    if (parts.length >= 2 && parts[0] === "agent") {
      return parts[1];
    }
  }

  // Fallback: use workspace directory name
  if (workspaceDir) {
    const dirName = workspaceDir.split("/").filter(Boolean).pop();
    if (dirName && dirName !== "clawd" && dirName !== "workspace") {
      return dirName;
    }
  }

  return "main";
}

/**
 * Execute kernle load and return output
 */
async function loadKernleMemory(stackId: string): Promise<string | null> {
  try {
    const { stdout } = await execAsync(`kernle -s ${stackId} load`, {
      timeout: 5000, // 5 second timeout
      maxBuffer: 1024 * 1024, // 1MB max output
    });

    return stdout.trim();
  } catch (error: any) {
    // Kernle not installed, stack doesn't exist, or command failed
    const stderr = error.stderr || error.message || "";

    // Only log if it's not a "not found" error
    if (!stderr.includes("command not found") && !stderr.includes("No stack found")) {
      console.warn(`[kernle-load] Failed to load memory for stack '${stackId}':`, stderr);
    }

    return null;
  }
}

/**
 * Hook handler: inject Kernle memory into bootstrap files
 */
const kernleLoadHook: HookHandler = async (event) => {
  // Only handle agent bootstrap events
  if (event.type !== "agent" || event.action !== "bootstrap") {
    return;
  }

  const context = event.context as BootstrapContext;

  // Ensure we have bootstrap files array
  if (!context.bootstrapFiles) {
    return;
  }

  const { workspaceDir, sessionKey } = context;
  const stackId = extractStackId(sessionKey, workspaceDir);

  // Load Kernle memory
  const memoryContent = await loadKernleMemory(stackId);

  if (!memoryContent) {
    // Kernle not available or no memory for this stack - continue without it
    return;
  }

  // Check if KERNLE.md already exists (avoid duplicates)
  const hasKernle = context.bootstrapFiles.some(
    (file) => file.path === "KERNLE.md" || file.path.endsWith("/KERNLE.md")
  );

  if (hasKernle) {
    return; // Already loaded
  }

  // Inject as virtual bootstrap file
  context.bootstrapFiles.push({
    path: "KERNLE.md",
    content: memoryContent,
    virtual: true,
  });
};

export default kernleLoadHook;
