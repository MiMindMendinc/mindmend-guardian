import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { DEFAULT_SETTINGS } from "./constants";
import type { ChatMessage, MemoryNote, ProjectCard, ProjectStatus, Settings, View } from "./types";
import { readJson, uid, writeJson } from "./utils/storage";
import { dateLabel, modelShort, timeLabel } from "./utils/format";

const views: { id: View; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "✦" },
  { id: "tools", label: "Tools", icon: "⌘" },
  { id: "memory", label: "Memory", icon: "◈" },
  { id: "projects", label: "Projects", icon: "▣" },
  { id: "settings", label: "Settings", icon: "⚙" },
  { id: "about", label: "About", icon: "A" },
];

const projectStatuses: ProjectStatus[] = ["Idea", "Building", "Testing", "Shipped", "Paused"];

function safeInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return invoke<T>(command, args).catch((error) => {
    throw new Error(typeof error === "string" ? error : "Ani backend command failed.");
  });
}

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState<Settings>(() => readJson("ani.settings", DEFAULT_SETTINGS));
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    readJson("ani.messages", [
      { id: uid("msg"), role: "ani", text: "Ani online. Local-first mode active. What are we building today?", createdAt: new Date().toISOString() },
    ])
  );
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ollama, setOllama] = useState("Checking...");
  const [tool, setTool] = useState("dns");
  const [target, setTarget] = useState("");
  const [toolResult, setToolResult] = useState("");
  const [notes, setNotes] = useState<MemoryNote[]>(() => readJson("ani.notes", []));
  const [noteText, setNoteText] = useState("");
  const [projects, setProjects] = useState<ProjectCard[]>(() => readJson("ani.projects", []));
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => writeJson("ani.settings", settings), [settings]);
  useEffect(() => writeJson("ani.messages", messages), [messages]);
  useEffect(() => writeJson("ani.notes", notes), [notes]);
  useEffect(() => writeJson("ani.projects", projects), [projects]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  useEffect(() => {
    safeInvoke<string>("check_ollama")
      .then((status) => setOllama(status))
      .catch(() => setOllama("offline"));
  }, []);

  const stats = useMemo(() => ({ notes: notes.length, projects: projects.length }), [notes.length, projects.length]);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;
    const now = new Date().toISOString();
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { id: uid("msg"), role: "user", text, createdAt: now }]);
    try {
      const reply = await safeInvoke<string>("ask_ani", { message: text, model: settings.model });
      setMessages((prev) => [...prev, { id: uid("msg"), role: "ani", text: reply, createdAt: new Date().toISOString() }]);
    } catch (error) {
      setMessages((prev) => [...prev, { id: uid("msg"), role: "system", text: error instanceof Error ? error.message : "Ani hit an error.", createdAt: new Date().toISOString() }]);
    } finally {
      setLoading(false);
    }
  }

  async function runTool() {
    const clean = target.trim();
    if (!clean) return;
    setToolResult("Running local safe tool...");
    try {
      setToolResult(await safeInvoke<string>("run_tool", { tool, target: clean }));
    } catch (error) {
      setToolResult(error instanceof Error ? error.message : "Tool failed.");
    }
  }

  function addNote() {
    if (!noteText.trim()) return;
    setNotes((prev) => [{ id: uid("note"), text: noteText.trim(), createdAt: new Date().toISOString() }, ...prev]);
    setNoteText("");
  }

  function addProject() {
    if (!projectName.trim()) return;
    setProjects((prev) => [{ id: uid("project"), name: projectName.trim(), description: projectDescription.trim(), status: "Idea", createdAt: new Date().toISOString() }, ...prev]);
    setProjectName("");
    setProjectDescription("");
  }

  const isOnline = ollama.toLowerCase().includes("online");

  return (
    <main className={`app ${settings.compactMode ? "compact" : ""}`}>
      <aside className="sidebar">
        <div className="brand"><div className="brandMark">A</div><div><strong>Ani 2027</strong><span>MindMend Local AI</span></div></div>
        <nav className="navList">{views.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}><span>{item.icon}</span>{item.label}</button>)}</nav>
        <div className="sidebarCard"><div className="statusLine"><i className="ok" />Local Mode</div><div className="muted">Cloud Sync: Disabled</div><div className="muted">External Upload: Disabled</div></div>
      </aside>

      <section className="workspace">
        <header className="topbar"><div><h1>{view === "chat" ? "Mission Control" : views.find((v) => v.id === view)?.label}</h1><p>Your machine. Your data. Your AI.</p></div><div className="topBadges"><span className={isOnline ? "badge online" : "badge offline"}>{isOnline ? "Ollama Online" : "Ollama Offline"}</span><span className="badge">{modelShort(settings.model)}</span><span className="badge">Local-only</span></div></header>

        {view === "chat" && <section className="panel chatPanel"><div className="missionGrid"><div className="metric"><span>Ollama</span><strong>{isOnline ? "Online" : "Offline"}</strong></div><div className="metric"><span>Model</span><strong>{modelShort(settings.model)}</strong></div><div className="metric"><span>Memory</span><strong>{stats.notes} notes</strong></div><div className="metric"><span>Projects</span><strong>{stats.projects}</strong></div></div><div className="messages">{messages.map((m) => <article key={m.id} className={`message ${m.role}`}><div className="bubble">{settings.showTimestamps && <time>{timeLabel(m.createdAt)}</time>}<p>{m.text}</p></div></article>)}{loading && <article className="message ani"><div className="bubble"><p>Ani is thinking locally...</p></div></article>}<div ref={chatEndRef} /></div><div className="composer"><textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendMessage(); } }} placeholder="Ask Ani something local..." /><button onClick={() => void sendMessage()} disabled={loading}>Send</button></div><div className="row"><button className="ghost" onClick={() => setMessages([])}>Clear Chat</button><button className="ghost" onClick={() => navigator.clipboard.writeText(messages[messages.length - 1]?.text ?? "")}>Copy Last</button></div></section>}

        {view === "tools" && <section className="panel"><h2>Safe Defensive Tools</h2><p className="muted">Authorization-first helpers only. No exploitation, brute force, or intrusive scans.</p><div className="toolGrid">{["dns", "recon", "scan", "whois", "linksentinel", "prompt_forge", "safety_review", "project_summary"].map((name) => <button key={name} className={tool === name ? "tool activeTool" : "tool"} onClick={() => setTool(name)}>{name.replace("_", " ")}</button>)}</div><div className="formRow"><input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="Domain, idea, prompt, or project text..." /><button onClick={() => void runTool()}>Run Tool</button></div><pre className="resultBox">{toolResult || "Tool output will appear here."}</pre></section>}

        {view === "memory" && <section className="panel"><h2>Local Memory</h2><p className="muted">Local browser storage only. No cloud sync.</p><div className="formRow"><input value={noteText} onChange={(e) => setNoteText(e.target.value)} placeholder="Save a local note..." /><button onClick={addNote}>Add Note</button></div><div className="cards">{notes.map((n) => <article className="card" key={n.id}><small>{dateLabel(n.createdAt)}</small><p>{n.text}</p><button className="ghost" onClick={() => setNotes((prev) => prev.filter((x) => x.id !== n.id))}>Delete</button></article>)}</div><button className="ghost" onClick={() => setNotes([])}>Clear All Notes</button></section>}

        {view === "projects" && <section className="panel"><h2>Project Workspace</h2><p className="muted">Track local AI builds, demos, pitches, and shipped work.</p><div className="projectForm"><input value={projectName} onChange={(e) => setProjectName(e.target.value)} placeholder="Project name" /><input value={projectDescription} onChange={(e) => setProjectDescription(e.target.value)} placeholder="Description" /><button onClick={addProject}>Add Project</button></div><div className="cards">{projects.map((p) => <article className="card" key={p.id}><small>{dateLabel(p.createdAt)}</small><h3>{p.name}</h3><p>{p.description}</p><select value={p.status} onChange={(e) => setProjects((prev) => prev.map((x) => x.id === p.id ? { ...x, status: e.target.value as ProjectStatus } : x))}>{projectStatuses.map((s) => <option key={s}>{s}</option>)}</select><button className="ghost" onClick={() => setProjects((prev) => prev.filter((x) => x.id !== p.id))}>Delete</button></article>)}</div></section>}

        {view === "settings" && <section className="panel"><h2>Settings</h2><label>Model name</label><input value={settings.model} onChange={(e) => setSettings({ ...settings, model: e.target.value })} /><div className="row"><button onClick={() => setSettings({ ...settings, model: DEFAULT_SETTINGS.model })}>Reset Model</button><button className="ghost" onClick={() => setSettings({ ...settings, compactMode: !settings.compactMode })}>Compact: {settings.compactMode ? "On" : "Off"}</button><button className="ghost" onClick={() => setSettings({ ...settings, showTimestamps: !settings.showTimestamps })}>Timestamps: {settings.showTimestamps ? "On" : "Off"}</button></div><div className="privacyCard">Privacy status: local-first, no app cloud sync, no external upload by default.</div></section>}

        {view === "about" && <section className="panel about"><div className="heroMark">A</div><h2>Ani 2027</h2><p>Ani 2027 is a local-first AI desktop assistant by Michigan MindMend Inc. It is designed for private local AI work, project building, safe diagnostics, writing, coding support, family-first workflows, and community-first AI.</p><h3>Your machine. Your data. Your control.</h3><div className="missionGrid"><div className="metric"><span>Version</span><strong>2027.0.0</strong></div><div className="metric"><span>Mode</span><strong>Local-first</strong></div><div className="metric"><span>Cloud Sync</span><strong>Disabled</strong></div></div></section>}
      </section>
    </main>
  );
}
