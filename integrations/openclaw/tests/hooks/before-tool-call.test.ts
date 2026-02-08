import { describe, it, expect, vi, beforeEach } from "vitest";
import { createBeforeToolCall } from "../../src/hooks/before-tool-call.js";

vi.mock("../../src/services/kernle-bridge.js", () => ({
  KernleBridge: vi.fn().mockImplementation(() => ({
    raw: vi.fn(),
  })),
}));

import { KernleBridge } from "../../src/services/kernle-bridge.js";

describe("before_tool_call hook", () => {
  let mockRaw: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockRaw = vi.fn().mockResolvedValue(true);
    vi.mocked(KernleBridge).mockImplementation(
      () => ({ raw: mockRaw } as any)
    );
  });

  it("blocks write_file to memory/", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "write_file",
      arguments: { file_path: "memory/2026-02-07.md", content: "today I learned..." },
    });

    expect(result).toEqual({
      blocked: true,
      message: expect.stringContaining("handled by Kernle"),
    });
  });

  it("blocks write_file to MEMORY.md", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "write_file",
      arguments: { file_path: "MEMORY.md", content: "updated memory" },
    });

    expect(result?.blocked).toBe(true);
  });

  it("blocks edit_file to memory paths", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "edit_file",
      arguments: { file_path: "memory/notes.md", new_string: "new content" },
    });

    expect(result?.blocked).toBe(true);
  });

  it("blocks create_file to memory paths", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "create_file",
      arguments: { file_path: "memory/new.md", content: "new file" },
    });

    expect(result?.blocked).toBe(true);
  });

  it("allows write_file to non-memory paths", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "write_file",
      arguments: { file_path: "src/index.ts", content: "code" },
    });

    expect(result).toBeUndefined();
  });

  it("allows non-write tools", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "read_file",
      arguments: { file_path: "memory/notes.md" },
    });

    expect(result).toBeUndefined();
  });

  it("captures content as raw entry before blocking", async () => {
    const hook = createBeforeToolCall({}, "test");

    await hook({
      name: "write_file",
      arguments: { file_path: "memory/daily.md", content: "important note" },
    });

    expect(mockRaw).toHaveBeenCalledWith(
      "test",
      expect.stringContaining("[memory-capture] memory/daily.md")
    );
    expect(mockRaw).toHaveBeenCalledWith(
      "test",
      expect.stringContaining("important note")
    );
  });

  it("truncates large captured content", async () => {
    const hook = createBeforeToolCall({}, "test");
    const longContent = "x".repeat(3000);

    await hook({
      name: "write_file",
      arguments: { file_path: "MEMORY.md", content: longContent },
    });

    const rawContent = mockRaw.mock.calls[0][1] as string;
    expect(rawContent).toContain("[truncated]");
    expect(rawContent.length).toBeLessThan(3000);
  });

  it("handles nested memory paths", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "write_file",
      arguments: { file_path: "/home/user/project/memory/daily.md", content: "note" },
    });

    expect(result?.blocked).toBe(true);
  });

  it("uses path argument as fallback", async () => {
    const hook = createBeforeToolCall({}, "test");

    const result = await hook({
      name: "write_file",
      arguments: { path: "memory/test.md", content: "note" },
    });

    expect(result?.blocked).toBe(true);
  });
});
