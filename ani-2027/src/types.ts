export type View = "chat" | "tools" | "memory" | "projects" | "settings" | "about";

export type Role = "user" | "ani" | "system";

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  createdAt: string;
}

export interface MemoryNote {
  id: string;
  text: string;
  createdAt: string;
}

export type ProjectStatus = "Idea" | "Building" | "Testing" | "Shipped" | "Paused";

export interface ProjectCard {
  id: string;
  name: string;
  description: string;
  status: ProjectStatus;
  createdAt: string;
}

export interface Settings {
  model: string;
  compactMode: boolean;
  showTimestamps: boolean;
}
