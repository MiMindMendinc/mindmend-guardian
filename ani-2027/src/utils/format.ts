export function timeLabel(value: string): string {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function dateLabel(value: string): string {
  return new Date(value).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

export function modelShort(model: string): string {
  return model.trim() || "llama3.2:3b";
}
