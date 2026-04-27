# Ani 2027 — MindMend Local AI Command Center

**Your machine. Your data. Your AI.**

Ani 2027 is a privacy-first local desktop AI command center built for Michigan MindMend Inc. It runs as a Tauri v2 desktop app, talks to a local Ollama model, and gives builders a clean workspace for chat, safe tools, memory notes, projects, and local-first AI workflows.

This is an active prototype intended to be honest, buildable, and easy to improve.

---

## What It Does

- Local AI chat through Ollama
- Ollama status check through `http://localhost:11434/api/tags`
- Default model: `llama3.2:3b`
- Dark Phalanx-inspired desktop interface
- Sidebar navigation: Chat, Tools, Memory, Projects, Settings, About
- Local-only memory notes using browser localStorage
- Local-only project cards using browser localStorage
- Defensive tools only: DNS lookup, safe recon checklist, safety review, prompt forge, project summary, LinkSentinel local placeholder
- No cloud sync by default
- No telemetry
- No intrusive scanning
- No credential collection
- No exploit or malware logic

---

## Tech Stack

- Tauri v2
- React 18
- TypeScript
- Vite
- Rust
- Reqwest
- Ollama local API

---

## Requirements

Install these first:

- Node.js 18+
- Rust stable
- Tauri prerequisites for your operating system
- Ollama

For Windows Tauri setup, follow the official Tauri prerequisites for WebView2, Microsoft C++ Build Tools, and Rust.

---

## Ollama Setup

Install the default model:

```bash
ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

Ani expects Ollama at:

```text
http://localhost:11434
```

---

## Run in Development

From this folder:

```bash
cd ani-2027
npm install
npm run tauri dev
```

---

## Build the Windows App

```bash
cd ani-2027
npm install
npm run tauri build
```

The Windows installer should appear under a Tauri output folder similar to:

```text
ani-2027/src-tauri/target/release/bundle/nsis/
```

The exact folder can vary depending on platform and Tauri version.

---

## Project Structure

```text
ani-2027/
├── package.json
├── index.html
├── tsconfig.json
├── vite.config.ts
├── README.md
├── .gitignore
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── styles.css
│   ├── types.ts
│   ├── constants.ts
│   └── utils/
│       ├── format.ts
│       └── storage.ts
└── src-tauri/
    ├── tauri.conf.json
    ├── Cargo.toml
    ├── build.rs
    └── src/
        ├── main.rs
        └── lib.rs
```

---

## Safety Boundaries

Ani's tool system is defensive and local-first.

Ani does **not** perform:

- exploitation
- brute force
- credential checks
- malware generation
- intrusive scanning
- hidden shell execution
- unauthorized cyber activity

The current DNS tool performs basic resolution only. Recon and scan tools return safe authorization-first checklists.

---

## Current Status

**Prototype / early flagship build.**

The UI, localStorage memory/projects, Tauri command wiring, Ollama chat path, Ollama status check, and safe tool backend are implemented. This repo still needs local build verification on a Windows machine with Node, Rust, Tauri prerequisites, and Ollama installed.

---

## Roadmap

- Add streaming Ollama responses
- Add model picker from installed Ollama tags
- Add encrypted local memory storage
- Add export/import for projects and notes
- Add GitHub Actions build check
- Add signed Windows installer flow
- Add screenshots and demo video
- Add dedicated repository once the prototype is stable

---

## Built By

Michigan MindMend Inc.  
Owosso, Michigan

Privacy-first, offline-capable AI tools for families, builders, nonprofits, and communities.
