import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const DEFAULT_TIMEOUT = 5000;
const MAX_BUFFER = 1024 * 1024; // 1MB
const KERNLE_BIN_PATTERN = /^[A-Za-z0-9._/@:+\\-]+$/;

export interface BridgeOptions {
  kernleBin?: string;
  timeout?: number;
}

/**
 * Shell exec wrapper for the `kernle` CLI.
 *
 * All calls have configurable timeout, 1MB max buffer, and return null on failure.
 */
export class KernleBridge {
  private bin: string;
  private timeout: number;

  constructor(options: BridgeOptions = {}) {
    this.bin = this.validateKernleBin(options.kernleBin ?? "kernle");
    this.timeout = this.validateTimeout(options.timeout ?? DEFAULT_TIMEOUT);
  }

  /**
   * Load memory for a stack. Returns the formatted memory output or null on failure.
   */
  async load(stackId: string, budget?: number): Promise<string | null> {
    const args = ["-s", stackId, "load"];
    if (budget !== undefined) {
      args.push("--budget", String(budget));
    }
    return this.exec(args);
  }

  /**
   * Save a checkpoint. Returns true on success.
   */
  async checkpoint(stackId: string, summary: string, context?: string): Promise<boolean> {
    const args = ["-s", stackId, "checkpoint", "save", summary];
    if (context !== undefined) {
      args.push("--context", context);
    }
    const result = await this.exec(args);
    return result !== null;
  }

  /**
   * Save a raw entry. Returns true on success.
   */
  async raw(stackId: string, content: string): Promise<boolean> {
    const result = await this.exec(["-s", stackId, "raw", content]);
    return result !== null;
  }

  /**
   * Search memory. Returns search results or null on failure.
   */
  async search(stackId: string, query: string): Promise<string | null> {
    return this.exec(["-s", stackId, "search", query]);
  }

  /**
   * Get stack status. Returns status output or null on failure.
   */
  async status(stackId: string): Promise<string | null> {
    return this.exec(["-s", stackId, "status"]);
  }

  private async exec(args: string[]): Promise<string | null> {
    try {
      const { stdout } = await execFileAsync(this.bin, args, {
        timeout: this.timeout,
        maxBuffer: MAX_BUFFER,
      });
      return stdout.trim();
    } catch (error: unknown) {
      const err = error as { stderr?: string; message?: string };
      const stderr = err.stderr ?? err.message ?? "";

      // Only warn for unexpected errors (not "not found" or "no stack")
      if (
        !stderr.includes("command not found") &&
        !stderr.includes("No stack found")
      ) {
        console.warn(`[kernle] CLI error: ${stderr.slice(0, 200)}`);
      }

      return null;
    }
  }

  private validateKernleBin(bin: string): string {
    if (!bin || !bin.trim()) {
      throw new Error("Invalid kernleBin: expected a non-empty executable path");
    }
    if (!KERNLE_BIN_PATTERN.test(bin)) {
      throw new Error("Invalid kernleBin: only executable paths are allowed");
    }
    return bin;
  }

  private validateTimeout(timeout: number): number {
    if (!Number.isInteger(timeout) || timeout <= 0) {
      throw new Error("Invalid timeout: expected a positive integer");
    }
    return timeout;
  }
}
