import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { resolveStackId } from "../src/stack-id.js";

describe("resolveStackId", () => {
  const originalEnv = process.env.KERNLE_STACK_ID;

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env.KERNLE_STACK_ID;
    } else {
      process.env.KERNLE_STACK_ID = originalEnv;
    }
  });

  it("uses explicit config stackId first", () => {
    process.env.KERNLE_STACK_ID = "env-stack";
    const result = resolveStackId(
      { stackId: "config-stack" },
      { sessionKey: "agent:session:main" }
    );
    expect(result).toBe("config-stack");
  });

  it("falls back to KERNLE_STACK_ID env var", () => {
    process.env.KERNLE_STACK_ID = "env-stack";
    const result = resolveStackId({}, { sessionKey: "agent:session:main" });
    expect(result).toBe("env-stack");
  });

  it("extracts agent ID from session key", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, { sessionKey: "agent:ash:main" });
    expect(result).toBe("ash");
  });

  it("handles session key with only two parts", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, { sessionKey: "agent:bob" });
    expect(result).toBe("bob");
  });

  it("falls back to workspace directory name", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, { workspaceDir: "/Users/ash/my-project" });
    expect(result).toBe("my-project");
  });

  it("skips generic workspace directory names", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, { workspaceDir: "/home/user/workspace" });
    expect(result).toBe("main");
  });

  it("falls back to 'main' when nothing available", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, {});
    expect(result).toBe("main");
  });

  it("ignores non-agent session keys", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId({}, { sessionKey: "system:init" });
    // Should not extract "init" since prefix is not "agent"
    expect(result).toBe("main");
  });

  it("prefers session key over workspace dir", () => {
    delete process.env.KERNLE_STACK_ID;
    const result = resolveStackId(
      {},
      { sessionKey: "agent:claire:work", workspaceDir: "/Users/x/other-project" }
    );
    expect(result).toBe("claire");
  });
});
