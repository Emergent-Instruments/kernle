import { exec } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

const DEFAULT_TIMEOUT = 5000;
const MAX_BUFFER = 1024 * 1024; // 1MB

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
    this.bin = options.kernleBin ?? "kernle";
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
  }

  /**
   * Load memory for a stack. Returns the formatted memory output or null on failure.
   */
  async load(stackId: string, budget?: number): Promise<string | null> {
    const budgetArg = budget ? ` --budget ${budget}` : "";
    return this.exec(`-s ${this.escape(stackId)} load${budgetArg}`);
  }

  /**
   * Save a checkpoint. Returns true on success.
   */
  async checkpoint(stackId: string, summary: string, context?: string): Promise<boolean> {
    const contextArg = context ? ` --context ${this.escape(context)}` : "";
    const result = await this.exec(
      `-s ${this.escape(stackId)} checkpoint save ${this.escape(summary)}${contextArg}`
    );
    return result !== null;
  }

  /**
   * Save a raw entry. Returns true on success.
   */
  async raw(stackId: string, content: string): Promise<boolean> {
    const result = await this.exec(
      `-s ${this.escape(stackId)} raw ${this.escape(content)}`
    );
    return result !== null;
  }

  /**
   * Search memory. Returns search results or null on failure.
   */
  async search(stackId: string, query: string): Promise<string | null> {
    return this.exec(`-s ${this.escape(stackId)} search ${this.escape(query)}`);
  }

  /**
   * Get stack status. Returns status output or null on failure.
   */
  async status(stackId: string): Promise<string | null> {
    return this.exec(`-s ${this.escape(stackId)} status`);
  }

  private async exec(args: string): Promise<string | null> {
    try {
      const { stdout } = await execAsync(`${this.bin} ${args}`, {
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

  /** Shell-escape a string argument. */
  private escape(value: string): string {
    // Wrap in single quotes, escaping any existing single quotes
    return `'${value.replace(/'/g, "'\\''")}'`;
  }
}
