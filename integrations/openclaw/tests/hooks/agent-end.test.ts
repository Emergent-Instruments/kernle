import { describe, it, expect, vi, beforeEach } from "vitest";
import { createAgentEnd } from "../../src/hooks/agent-end.js";

vi.mock("../../src/services/kernle-bridge.js", () => ({
  KernleBridge: vi.fn().mockImplementation(() => ({
    checkpoint: vi.fn(),
    raw: vi.fn(),
  })),
}));

import { KernleBridge } from "../../src/services/kernle-bridge.js";

describe("agent_end hook", () => {
  let mockCheckpoint: ReturnType<typeof vi.fn>;
  let mockRaw: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCheckpoint = vi.fn().mockResolvedValue(true);
    mockRaw = vi.fn().mockResolvedValue(true);
    vi.mocked(KernleBridge).mockImplementation(
      () => ({ checkpoint: mockCheckpoint, raw: mockRaw } as any)
    );
  });

  it("saves checkpoint with last user message as task", async () => {
    const hook = createAgentEnd({ stackId: "test" });

    await hook({
      messages: [
        { role: "user", content: "Fix the login bug" },
        { role: "assistant", content: "I fixed the login bug by..." },
      ],
    });

    expect(mockCheckpoint).toHaveBeenCalledWith(
      "test",
      "Fix the login bug",
      "I fixed the login bug by..."
    );
  });

  it("saves raw entry marking session end", async () => {
    const hook = createAgentEnd({ stackId: "test" });

    await hook({
      messages: [{ role: "user", content: "Do something" }],
    });

    expect(mockRaw).toHaveBeenCalledWith(
      "test",
      "Session ended. Task: Do something"
    );
  });

  it("uses fallback text when no messages", async () => {
    const hook = createAgentEnd({ stackId: "test" });

    await hook({});

    expect(mockCheckpoint).toHaveBeenCalledWith(
      "test",
      "Session ended",
      undefined
    );
  });

  it("truncates long user messages", async () => {
    const hook = createAgentEnd({ stackId: "test" });
    const longMessage = "x".repeat(300);

    await hook({
      messages: [{ role: "user", content: longMessage }],
    });

    const taskArg = mockCheckpoint.mock.calls[0][1] as string;
    expect(taskArg.length).toBe(200);
    expect(taskArg.endsWith("...")).toBe(true);
  });

  it("truncates long assistant messages", async () => {
    const hook = createAgentEnd({ stackId: "test" });
    const longMessage = "y".repeat(600);

    await hook({
      messages: [
        { role: "user", content: "task" },
        { role: "assistant", content: longMessage },
      ],
    });

    const contextArg = mockCheckpoint.mock.calls[0][2] as string;
    expect(contextArg.length).toBe(500);
    expect(contextArg.endsWith("...")).toBe(true);
  });

  it("finds the last user message (not first)", async () => {
    const hook = createAgentEnd({ stackId: "test" });

    await hook({
      messages: [
        { role: "user", content: "First question" },
        { role: "assistant", content: "First answer" },
        { role: "user", content: "Follow-up question" },
        { role: "assistant", content: "Follow-up answer" },
      ],
    });

    expect(mockCheckpoint).toHaveBeenCalledWith(
      "test",
      "Follow-up question",
      "Follow-up answer"
    );
  });

  it("continues even if checkpoint fails", async () => {
    mockCheckpoint.mockRejectedValue(new Error("fail"));
    const hook = createAgentEnd({ stackId: "test" });

    // Should not throw
    await hook({
      messages: [{ role: "user", content: "task" }],
    });

    expect(mockRaw).toHaveBeenCalled();
  });
});
