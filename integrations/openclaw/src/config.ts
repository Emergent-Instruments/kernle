import type { PluginConfig } from "./types.js";

const MAX_TOKEN_BUDGET = 50000;
const MAX_TIMEOUT_MS = 120000;
const STACK_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const KERNLE_BIN_PATTERN = /^[A-Za-z0-9._/@:+\\-]+$/;

export function validatePluginConfig(config: PluginConfig): PluginConfig {
  if (config.stackId !== undefined && !STACK_ID_PATTERN.test(config.stackId)) {
    throw new Error(
      "Invalid plugin config: stackId must be 1-128 chars and use [A-Za-z0-9._:-]"
    );
  }

  if (
    config.tokenBudget !== undefined &&
    (!Number.isInteger(config.tokenBudget) ||
      config.tokenBudget < 1 ||
      config.tokenBudget > MAX_TOKEN_BUDGET)
  ) {
    throw new Error(
      `Invalid plugin config: tokenBudget must be an integer between 1 and ${MAX_TOKEN_BUDGET}`
    );
  }

  if (
    config.timeout !== undefined &&
    (!Number.isInteger(config.timeout) || config.timeout < 1 || config.timeout > MAX_TIMEOUT_MS)
  ) {
    throw new Error(
      `Invalid plugin config: timeout must be an integer between 1 and ${MAX_TIMEOUT_MS}`
    );
  }

  if (config.kernleBin !== undefined) {
    if (!config.kernleBin.trim()) {
      throw new Error("Invalid plugin config: kernleBin must be a non-empty path");
    }
    if (!KERNLE_BIN_PATTERN.test(config.kernleBin)) {
      throw new Error("Invalid plugin config: kernleBin must be an executable path only");
    }
  }

  return config;
}
