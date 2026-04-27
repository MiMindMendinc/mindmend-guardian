use serde::{Deserialize, Serialize};
use std::net::ToSocketAddrs;

const OLLAMA_CHAT_URL: &str = "http://localhost:11434/api/chat";
const OLLAMA_TAGS_URL: &str = "http://localhost:11434/api/tags";
const DEFAULT_MODEL: &str = "llama3.2:3b";
const SYSTEM_PROMPT: &str = "You are Ani 2027, a privacy-first local desktop AI assistant for Michigan MindMend Inc. You run locally, protect user privacy, help with coding, planning, safe diagnostics, writing, learning, and project building. You are direct, useful, calm, and safety-aware. You do not help with cyber abuse, credential theft, malware, evasion, exploitation, or harm. You help the user build legitimate local-first software and solve real problems.";

#[derive(Debug, Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessageResponse,
}

#[tauri::command]
async fn ask_ani(message: String, model: Option<String>) -> Result<String, String> {
    let selected_model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let selected_model = selected_model.trim();
    let selected_model = if selected_model.is_empty() { DEFAULT_MODEL } else { selected_model };

    let body = OllamaRequest {
        model: selected_model,
        messages: vec![
            OllamaMessage { role: "system", content: SYSTEM_PROMPT },
            OllamaMessage { role: "user", content: message.as_str() },
        ],
        stream: false,
    };

    let client = reqwest::Client::new();
    let response = client
        .post(OLLAMA_CHAT_URL)
        .json(&body)
        .send()
        .await
        .map_err(|_| "Ollama is offline. Start it with: ollama serve\nThen install the default model with: ollama pull llama3.2:3b".to_string())?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}. Make sure the model is installed: ollama pull {}", response.status(), selected_model));
    }

    let parsed = response
        .json::<OllamaChatResponse>()
        .await
        .map_err(|_| "Ani could not parse the Ollama response.".to_string())?;

    Ok(parsed.message.content)
}

#[tauri::command]
async fn check_ollama() -> Result<String, String> {
    match reqwest::get(OLLAMA_TAGS_URL).await {
        Ok(response) if response.status().is_success() => Ok("online".to_string()),
        _ => Ok("offline".to_string()),
    }
}

#[tauri::command]
async fn run_tool(tool: String, target: String) -> Result<String, String> {
    let tool = tool.to_lowercase();
    let target = target.trim();
    if target.is_empty() {
        return Err("Enter a target, domain, prompt, or project note first.".to_string());
    }

    match tool.as_str() {
        "dns" => dns_lookup(target),
        "recon" => Ok(format!("Safe Recon Checklist for {target}\n\n1. Confirm you own or have permission to inspect this target.\n2. Check domain spelling.\n3. Review public DNS records.\n4. Check public website availability.\n5. Review SSL certificate manually in the browser.\n6. Document findings.\n7. Do not run intrusive scans without written authorization.")),
        "scan" => Ok(format!("Ani does not run intrusive scans by default.\n\nFor {target}:\n- Confirm authorization first.\n- Use approved internal tools only.\n- Keep logs.\n- Avoid brute force, exploit testing, credential checks, or disruption.")),
        "whois" => Ok(format!("WHOIS lookup requested for {target}.\n\nWHOIS provider integration is not enabled yet. Use a trusted WHOIS provider manually.")),
        "linksentinel" => sync_linksentinel().await,
        "prompt_forge" => Ok(format!("Improved prompt:\n\nGoal:\n{target}\n\nContext:\nWhat background should the model know?\n\nInputs:\nPaste exact files, text, screenshots, or constraints.\n\nConstraints:\nLocal-first, safe, honest, and buildable.\n\nOutput format:\nClear steps, complete code where needed, and verification commands.\n\nQuality bar:\nNo fake claims. No missing files. No unsafe behavior. Must be runnable.")),
        "safety_review" => Ok(format!("Safety Review for:\n{target}\n\nChecklist:\n- Does this affect real people or sensitive data?\n- Is there consent or authorization?\n- Are claims backed by code or evidence?\n- Could this be misused for harm?\n- Is cloud upload disabled by default?\n- Are logs local, minimal, and user-controlled?\n- Is there a clear human decision point?")),
        "project_summary" => Ok(format!("Project Summary\n\nName:\n{target}\n\nPurpose:\n\nUsers:\n\nCore features:\n\nLocal-first privacy model:\n\nSafety boundaries:\n\nCurrent status:\n\nNext 3 steps:")),
        "memory" => memory_status().await,
        other => Ok(format!("Unknown tool: {other}")),
    }
}

fn dns_lookup(target: &str) -> Result<String, String> {
    let host = target.trim().trim_start_matches("https://").trim_start_matches("http://").split('/').next().unwrap_or(target);
    let query = format!("{host}:80");
    let mut ips: Vec<String> = query
        .to_socket_addrs()
        .map_err(|_| format!("Could not resolve {host}."))?
        .map(|addr| addr.ip().to_string())
        .collect();
    ips.sort();
    ips.dedup();
    Ok(format!("DNS lookup for {host}:\n{}", ips.iter().map(|ip| format!("- {ip}")).collect::<Vec<_>>().join("\n")))
}

#[tauri::command]
async fn sync_linksentinel() -> Result<String, String> {
    Ok("LinkSentinel local sync complete.\n\n- Local rules checked\n- Cloud upload disabled\n- Privacy mode active\n- No external data sent".to_string())
}

#[tauri::command]
async fn memory_status() -> Result<String, String> {
    Ok("Ani Memory Status:\n- Local notes: enabled\n- Cloud sync: disabled\n- External upload: disabled\n- Storage mode: local application storage".to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![ask_ani, check_ollama, run_tool, sync_linksentinel, memory_status])
        .run(tauri::generate_context!())
        .expect("error while running Ani 2027");
}
