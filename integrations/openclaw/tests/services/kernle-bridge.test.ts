import { describe, it, expect, vi, beforeEach } from "vitest";
import { KernleBridge } from "../../src/services/kernle-bridge.js";

// Mock child_process.execFile
vi.mock("node:child_process", () => ({
  execFile: vi.fn(),
}));
vi.mock("node:util", () => ({
  promisify: function (fn: Function) { return fn; },
}));

import { execFile } from "node:child_process";
const mockExecFile = vi.mocked(execFile);

describe("KernleBridge", () => {
  let bridge: KernleBridge;

  beforeEach(() => {
    vi.clearAllMocks();
    bridge = new KernleBridge();
  });

  describe("load", () => {
    it("calls kernle load with stack ID", async () => {
      mockExecFile.mockResolvedValue({ stdout: "# Memory\nvalues...\n", stderr: "" } as any);

      const result = await bridge.load("my-project");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "my-project", "load"],
        expect.objectContaining({ timeout: 5000, maxBuffer: 1048576 })
      );
      expect(result).toBe("# Memory\nvalues...");
    });

    it("includes budget when provided", async () => {
      mockExecFile.mockResolvedValue({ stdout: "memory\n", stderr: "" } as any);

      await bridge.load("test", 4000);

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "test", "load", "--budget", "4000"],
        expect.any(Object)
      );
    });

    it("passes potentially unsafe stack IDs as argv, not shell text", async () => {
      mockExecFile.mockResolvedValue({ stdout: "memory\n", stderr: "" } as any);

      await bridge.load("project; rm -rf /", 1234);

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "project; rm -rf /", "load", "--budget", "1234"],
        expect.any(Object)
      );
    });

    it("returns null on command failure", async () => {
      mockExecFile.mockRejectedValue({ stderr: "command not found" });

      const result = await bridge.load("test");

      expect(result).toBeNull();
    });

    it("returns null on timeout", async () => {
      mockExecFile.mockRejectedValue({ message: "timeout" });

      const result = await bridge.load("test");

      expect(result).toBeNull();
    });
  });

  describe("checkpoint", () => {
    it("calls kernle checkpoint save", async () => {
      mockExecFile.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      const result = await bridge.checkpoint("proj", "task done");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "checkpoint", "save", "task done"],
        expect.any(Object)
      );
      expect(result).toBe(true);
    });

    it("includes context when provided", async () => {
      mockExecFile.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      await bridge.checkpoint("proj", "task", "extra context");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "checkpoint", "save", "task", "--context", "extra context"],
        expect.any(Object)
      );
    });

    it("returns false on failure", async () => {
      mockExecFile.mockRejectedValue({ stderr: "error" });

      const result = await bridge.checkpoint("proj", "task");

      expect(result).toBe(false);
    });
  });

  describe("raw", () => {
    it("calls kernle raw with content", async () => {
      mockExecFile.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      const result = await bridge.raw("proj", "quick thought");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "raw", "quick thought"],
        expect.any(Object)
      );
      expect(result).toBe(true);
    });

    it("preserves literal content as a single argv argument", async () => {
      mockExecFile.mockResolvedValue({ stdout: "saved\n", stderr: "" } as any);

      await bridge.raw("proj", "it's a test");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "raw", "it's a test"],
        expect.any(Object)
      );
    });
  });

  describe("search", () => {
    it("calls kernle search with query", async () => {
      mockExecFile.mockResolvedValue({ stdout: "result1\nresult2\n", stderr: "" } as any);

      const result = await bridge.search("proj", "topic");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "search", "topic"],
        expect.any(Object)
      );
      expect(result).toBe("result1\nresult2");
    });
  });

  describe("status", () => {
    it("calls kernle status", async () => {
      mockExecFile.mockResolvedValue({ stdout: "Stack: proj\nMemories: 42\n", stderr: "" } as any);

      const result = await bridge.status("proj");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "status"],
        expect.any(Object)
      );
      expect(result).toBe("Stack: proj\nMemories: 42");
    });
  });

  describe("custom options", () => {
    it("uses custom binary path", async () => {
      mockExecFile.mockResolvedValue({ stdout: "ok\n", stderr: "" } as any);
      const custom = new KernleBridge({ kernleBin: "/usr/local/bin/kernle" });

      await custom.status("proj");

      expect(mockExecFile).toHaveBeenCalledWith(
        "/usr/local/bin/kernle",
        ["-s", "proj", "status"],
        expect.any(Object)
      );
    });

    it("uses custom timeout", async () => {
      mockExecFile.mockResolvedValue({ stdout: "ok\n", stderr: "" } as any);
      const custom = new KernleBridge({ timeout: 10000 });

      await custom.status("proj");

      expect(mockExecFile).toHaveBeenCalledWith(
        "kernle",
        ["-s", "proj", "status"],
        expect.objectContaining({ timeout: 10000 })
      );
    });

    it("rejects invalid binary config", () => {
      expect(() => new KernleBridge({ kernleBin: "kernle --debug" })).toThrow(
        "Invalid kernleBin"
      );
    });

    it("rejects non-positive timeout config", () => {
      expect(() => new KernleBridge({ timeout: 0 })).toThrow("Invalid timeout");
    });
  });
});
