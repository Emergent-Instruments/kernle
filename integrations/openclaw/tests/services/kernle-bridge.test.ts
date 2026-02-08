import { describe, it, expect, vi, beforeEach } from "vitest";
import { KernleBridge } from "../../src/services/kernle-bridge.js";

// Mock child_process.exec
vi.mock("node:child_process", () => ({
  exec: vi.fn(),
}));
vi.mock("node:util", () => ({
  promisify: (fn: Function) => fn,
}));

import { exec } from "node:child_process";
const mockExec = vi.mocked(exec);

describe("KernleBridge", () => {
  let bridge: KernleBridge;

  beforeEach(() => {
    vi.clearAllMocks();
    bridge = new KernleBridge();
  });

  describe("load", () => {
    it("calls kernle load with stack ID", async () => {
      mockExec.mockResolvedValue({ stdout: "# Memory\nvalues...\n", stderr: "" } as any);

      const result = await bridge.load("my-project");

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'my-project' load",
        expect.objectContaining({ timeout: 5000, maxBuffer: 1048576 })
      );
      expect(result).toBe("# Memory\nvalues...");
    });

    it("includes budget when provided", async () => {
      mockExec.mockResolvedValue({ stdout: "memory\n", stderr: "" } as any);

      await bridge.load("test", 4000);

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'test' load --budget 4000",
        expect.any(Object)
      );
    });

    it("returns null on command failure", async () => {
      mockExec.mockRejectedValue({ stderr: "command not found" });

      const result = await bridge.load("test");

      expect(result).toBeNull();
    });

    it("returns null on timeout", async () => {
      mockExec.mockRejectedValue({ message: "timeout" });

      const result = await bridge.load("test");

      expect(result).toBeNull();
    });
  });

  describe("checkpoint", () => {
    it("calls kernle checkpoint save", async () => {
      mockExec.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      const result = await bridge.checkpoint("proj", "task done");

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'proj' checkpoint save 'task done'",
        expect.any(Object)
      );
      expect(result).toBe(true);
    });

    it("includes context when provided", async () => {
      mockExec.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      await bridge.checkpoint("proj", "task", "extra context");

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'proj' checkpoint save 'task' --context 'extra context'",
        expect.any(Object)
      );
    });

    it("returns false on failure", async () => {
      mockExec.mockRejectedValue({ stderr: "error" });

      const result = await bridge.checkpoint("proj", "task");

      expect(result).toBe(false);
    });
  });

  describe("raw", () => {
    it("calls kernle raw with content", async () => {
      mockExec.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      const result = await bridge.raw("proj", "quick thought");

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'proj' raw 'quick thought'",
        expect.any(Object)
      );
      expect(result).toBe(true);
    });

    it("escapes single quotes in content", async () => {
      mockExec.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      await bridge.raw("proj", "it's a test");

      expect(mockExec).toHaveBeenCalledWith(
        "kernle -s 'proj' raw 'it'\\''s a test'",
        expect.any(Object)
      );
    });
  });

  describe("search", () => {
    it("calls kernle search with query", async () => {
      mockExec.mockResolvedValue({ stdout: "result1\nresult2\n", stderr: "" } as any);

      const result = await bridge.search("proj", "topic");

      expect(result).toBe("result1\nresult2");
    });
  });

  describe("status", () => {
    it("calls kernle status", async () => {
      mockExec.mockResolvedValue({ stdout: "Stack: proj\nMemories: 42\n", stderr: "" } as any);

      const result = await bridge.status("proj");

      expect(result).toBe("Stack: proj\nMemories: 42");
    });
  });

  describe("custom options", () => {
    it("uses custom binary path", async () => {
      mockExec.mockResolvedValue({ stdout: "ok\n", stderr: "" } as any);
      const custom = new KernleBridge({ kernleBin: "/usr/local/bin/kernle" });

      await custom.status("proj");

      expect(mockExec).toHaveBeenCalledWith(
        "/usr/local/bin/kernle -s 'proj' status",
        expect.any(Object)
      );
    });

    it("uses custom timeout", async () => {
      mockExec.mockResolvedValue({ stdout: "ok\n", stderr: "" } as any);
      const custom = new KernleBridge({ timeout: 10000 });

      await custom.status("proj");

      expect(mockExec).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ timeout: 10000 })
      );
    });
  });
});
