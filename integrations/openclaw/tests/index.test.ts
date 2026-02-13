import { describe, it, expect, vi, beforeEach } from "vitest";
import { register } from "../src/index.js";

vi.mock("../src/services/kernle-bridge.js", () => ({
  KernleBridge: vi.fn().mockImplementation(function () {
    return { load: vi.fn(), checkpoint: vi.fn(), raw: vi.fn() };
  }),
}));

describe("register", () => {
  let mockApi: {
    on: ReturnType<typeof vi.fn>;
    getConfig: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockApi = {
      on: vi.fn(),
      getConfig: vi.fn().mockReturnValue({ stackId: "test" }),
    };
  });

  it("registers all four hooks", () => {
    register(mockApi as any);

    const registeredEvents = mockApi.on.mock.calls.map(
      (call: any[]) => call[0]
    );
    expect(registeredEvents).toContain("before_agent_start");
    expect(registeredEvents).toContain("agent_end");
    expect(registeredEvents).toContain("before_tool_call");
    expect(registeredEvents).toContain("tool_result_persist");
    expect(mockApi.on).toHaveBeenCalledTimes(4);
  });

  it("reads config from api", () => {
    register(mockApi as any);

    expect(mockApi.getConfig).toHaveBeenCalled();
  });

  it("rejects invalid kernleBin config", () => {
    mockApi.getConfig.mockReturnValue({ stackId: "test", kernleBin: "kernle --debug" });

    expect(() => register(mockApi as any)).toThrow("Invalid plugin config");
    expect(mockApi.on).not.toHaveBeenCalled();
  });

  it("rejects out-of-range timeout config", () => {
    mockApi.getConfig.mockReturnValue({ stackId: "test", timeout: 0 });

    expect(() => register(mockApi as any)).toThrow("Invalid plugin config");
    expect(mockApi.on).not.toHaveBeenCalled();
  });

  describe("tool_result_persist handler", () => {
    it("trims large kernle output", () => {
      register(mockApi as any);

      // Find the tool_result_persist handler
      const persistCall = mockApi.on.mock.calls.find(
        (call: any[]) => call[0] === "tool_result_persist"
      );
      const handler = persistCall![1];

      const longResult = "# Kernle Memory\n" + "x".repeat(3000);
      const result = handler({ toolName: "kernle-load", result: longResult });

      expect(result.result.length).toBeLessThan(longResult.length);
      expect(result.result).toContain("[Trimmed:");
    });

    it("does not trim short kernle output", () => {
      register(mockApi as any);

      const persistCall = mockApi.on.mock.calls.find(
        (call: any[]) => call[0] === "tool_result_persist"
      );
      const handler = persistCall![1];

      const result = handler({
        toolName: "kernle-load",
        result: "# Kernle Memory\nShort output",
      });

      expect(result).toBeUndefined();
    });

    it("ignores non-kernle tool output", () => {
      register(mockApi as any);

      const persistCall = mockApi.on.mock.calls.find(
        (call: any[]) => call[0] === "tool_result_persist"
      );
      const handler = persistCall![1];

      const longResult = "x".repeat(3000);
      const result = handler({ toolName: "other-tool", result: longResult });

      expect(result).toBeUndefined();
    });
  });
});
