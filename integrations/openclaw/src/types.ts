/**
 * OpenClaw Plugin SDK types.
 *
 * These define the contract between this plugin and the OpenClaw host.
 * Based on the OpenClaw Plugin SDK hooks that are currently wired and working.
 */

export interface PluginConfig {
  stackId?: string;
  tokenBudget?: number;
  timeout?: number;
  kernleBin?: string;
}

export interface PluginApi {
  on(event: "before_agent_start", handler: BeforeAgentStartHandler): void;
  on(event: "agent_end", handler: AgentEndHandler): void;
  on(event: "before_tool_call", handler: BeforeToolCallHandler): void;
  on(event: "tool_result_persist", handler: ToolResultPersistHandler): void;
  getConfig(): PluginConfig;
}

// --- before_agent_start ---

export interface BeforeAgentStartContext {
  sessionKey?: string;
  workspaceDir?: string;
  agentId?: string;
}

export interface BeforeAgentStartResult {
  prependContext?: string;
}

export type BeforeAgentStartHandler = (
  context: BeforeAgentStartContext
) => Promise<BeforeAgentStartResult | void>;

// --- agent_end ---

export interface ConversationMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface AgentEndContext {
  sessionKey?: string;
  workspaceDir?: string;
  messages?: ConversationMessage[];
}

export type AgentEndHandler = (context: AgentEndContext) => Promise<void>;

// --- before_tool_call ---

export interface ToolCallInfo {
  name: string;
  arguments: Record<string, unknown>;
}

export interface BeforeToolCallResult {
  blocked?: boolean;
  message?: string;
}

export type BeforeToolCallHandler = (
  tool: ToolCallInfo
) => Promise<BeforeToolCallResult | void>;

// --- tool_result_persist ---

export interface ToolResultInfo {
  toolName: string;
  result: string;
}

export interface ToolResultPersistResult {
  result?: string;
}

export type ToolResultPersistHandler = (
  info: ToolResultInfo
) => ToolResultPersistResult | void;
