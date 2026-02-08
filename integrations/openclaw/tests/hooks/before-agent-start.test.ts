import { describe, it, expect, vi, beforeEach } from "vitest";
import { createBeforeAgentStart } from "../../src/hooks/before-agent-start.js";

// Mock KernleBridge
vi.mock("../../src/services/kernle-bridge.js", () => ({
  KernleBridge: vi.fn().mockImplementation(() => ({
    load: vi.fn(),
  })),
}));

import { KernleBridge } from "../../src/services/kernle-bridge.js";

describe("before_agent_start hook", () => {
  let mockLoad: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockLoad = vi.fn();
    vi.mocked(KernleBridge).mockImplementation(
      () => ({ load: mockLoad } as any)
    );
  });

  it("loads memory and returns prependContext", async () => {
    mockLoad.mockResolvedValue("# Kernle Memory\n\nValues: be helpful");
    const hook = createBeforeAgentStart({ stackId: "test" });

    const result = await hook({ sessionKey: "agent:test:main" });

    expect(mockLoad).toHaveBeenCalledWith("test", 8000);
    expect(result).toEqual({
      prependContext: "# Kernle Memory\n\nValues: be helpful",
    });
  });

  it("returns void when load fails", async () => {
    mockLoad.mockResolvedValue(null);
    const hook = createBeforeAgentStart({ stackId: "test" });

    const result = await hook({ sessionKey: "agent:test:main" });

    expect(result).toBeUndefined();
  });

  it("uses configured token budget", async () => {
    mockLoad.mockResolvedValue("memory");
    const hook = createBeforeAgentStart({ stackId: "test", tokenBudget: 4000 });

    await hook({});

    expect(mockLoad).toHaveBeenCalledWith("test", 4000);
  });

  it("resolves stack ID from session key when not configured", async () => {
    mockLoad.mockResolvedValue("memory");
    const hook = createBeforeAgentStart({});

    await hook({ sessionKey: "agent:ash:main" });

    expect(mockLoad).toHaveBeenCalledWith("ash", 8000);
  });

  it("passes bridge options from config", () => {
    createBeforeAgentStart({
      kernleBin: "/custom/kernle",
      timeout: 10000,
    });

    expect(KernleBridge).toHaveBeenCalledWith({
      kernleBin: "/custom/kernle",
      timeout: 10000,
    });
  });
});
