import type { Settings } from "./types";

export const DEFAULT_MODEL = "llama3.2:3b";

export const DEFAULT_SETTINGS: Settings = {
  model: DEFAULT_MODEL,
  compactMode: false,
  showTimestamps: true,
};

export const SYSTEM_PROMPT = "You are Ani 2027, a privacy-first local desktop AI assistant for Michigan MindMend Inc. You run locally, protect user privacy, help with coding, planning, safe diagnostics, writing, learning, and project building. You are direct, useful, calm, and safety-aware.";
