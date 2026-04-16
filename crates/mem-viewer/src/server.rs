use axum::{
    extract::{Path, State},
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use mem_core::{Context, FactGraph};

#[derive(Clone)]
pub struct ViewerState {
    pub context: Arc<RwLock<Context>>,
    pub graph: Arc<FactGraph>,
}

pub async fn start_viewer(state: ViewerState, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/session/current", get(get_current_session))
        .route("/api/observations/:id", get(get_observation))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    tracing::info!("Web viewer silently started on http://0.0.0.0:{}", port);
    
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Ok(())
}

async fn get_current_session(State(state): State<ViewerState>) -> impl IntoResponse {
    let ctx = state.context.read().await;
    Json(json!({
        "items": ctx.items,
        "token_estimate": mem_core::estimate_tokens(&serde_json::to_string(&ctx.items).unwrap_or_default())
    }))
}

async fn get_observation(
    Path(id): Path<String>,
    State(state): State<ViewerState>,
) -> impl IntoResponse {
    if let Some(fact) = state.graph.get_fact(&id) {
        Json(json!({ "status": "success", "data": fact }))
    } else {
        Json(json!({ "status": "not_found", "data": null }))
    }
}

async fn index_handler() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindPalace Memory Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0f172a;
            --bg-surface: rgba(30, 41, 59, 0.7);
            --bg-glow: rgba(56, 189, 248, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-primary: #38bdf8;
            --accent-secondary: #c084fc;
            --border-subtle: rgba(255, 255, 255, 0.1);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Ambient Glow Background */
        body::before {
            content: '';
            position: fixed;
            top: -50%; left: -50%; right: -50%; bottom: -50%;
            background: radial-gradient(circle at 50% 50%, var(--bg-glow), transparent 60%);
            z-index: -1;
            animation: pulse-glow 15s ease-in-out infinite alternate;
        }

        @keyframes pulse-glow {
            0% { transform: scale(0.8) translate(5%, 5%); }
            100% { transform: scale(1.2) translate(-5%, -5%); }
        }

        header {
            padding: 2rem;
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: rgba(15, 23, 42, 0.8);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .stats-badge {
            background: var(--bg-surface);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            border: 1px solid var(--border-subtle);
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            gap: 1rem;
        }

        .stats-badge span.accent {
            color: var(--accent-primary);
        }

        main {
            flex: 1;
            padding: 3rem 2rem;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 2rem;
        }

        .glass-panel {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .glass-panel:hover {
            transform: translateY(-5px);
        }

        .glass-panel::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        }

        h2 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .context-stream {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: 600px;
            overflow-y: auto;
            padding-right: 1rem;
        }

        .context-stream::-webkit-scrollbar {
            width: 6px;
        }
        .context-stream::-webkit-scrollbar-thumb {
            background: var(--border-subtle);
            border-radius: 10px;
        }

        .message {
            padding: 1.25rem;
            border-radius: 16px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid transparent;
            transition: all 0.2s ease;
            animation: slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            opacity: 0;
            transform: translateY(20px);
        }

        @keyframes slide-up {
            to { opacity: 1; transform: translateY(0); }
        }

        .message:hover {
            border-color: var(--border-subtle);
            background: rgba(255, 255, 255, 0.03);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }

        .message-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .role-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 800;
        }

        .role-User { background: rgba(56, 189, 248, 0.2); color: var(--accent-primary); }
        .role-Assistant { background: rgba(192, 132, 252, 0.2); color: var(--accent-secondary); }
        .role-Tool { background: rgba(244, 63, 94, 0.2); color: #f43f5e; }

        .content {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        /* Sidebar Insights */
        .insight-card {
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-subtle);
        }

        .insight-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--accent-primary);
            margin: 0.5rem 0;
        }

        .metric-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Responsive */
        @media (max-width: 1024px) {
            main { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <header>
        <h1>MindPalace Telemetry</h1>
        <div class="stats-badge" id="globalStats">
            <span>Live Session</span>
            <span class="accent">• Connected</span>
        </div>
    </header>

    <main>
        <div class="glass-panel">
            <h2>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                Context Stream
            </h2>
            <div class="context-stream" id="streamContainer">
                <!-- Dynamically injected messages -->
                <div style="color: var(--text-muted); text-align: center; margin-top: 2rem;">Awaiting telemetry data...</div>
            </div>
        </div>

        <div class="glass-panel" style="background: rgba(15, 23, 42, 0.4);">
            <h2>Insights</h2>
            
            <div class="insight-card">
                <div class="metric-label">Estimated Token Load</div>
                <div class="insight-value" id="tokenCount">0</div>
                <div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-top: 1rem; overflow: hidden;">
                    <div id="tokenBar" style="height: 100%; width: 0%; background: var(--accent-primary); transition: width 0.5s ease;"></div>
                </div>
            </div>

            <div class="insight-card">
                <div class="metric-label">Memory Items</div>
                <div class="insight-value" id="itemCount" style="color: var(--accent-secondary);">0</div>
            </div>
        </div>
    </main>

    <script>
        async function fetchContext() {
            try {
                const res = await fetch('/api/session/current');
                const data = await res.json();
                
                // Update stats
                document.getElementById('tokenCount').innerText = data.token_estimate.toLocaleString();
                document.getElementById('itemCount').innerText = data.items.length;
                
                const percentage = Math.min((data.token_estimate / 128000) * 100, 100);
                document.getElementById('tokenBar').style.width = percentage + '%';

                // Update stream
                const container = document.getElementById('streamContainer');
                if (data.items.length > 0) {
                    container.innerHTML = '';
                    data.items.forEach((item, index) => {
                        const div = document.createElement('div');
                        div.className = 'message';
                        div.style.animationDelay = `${index * 0.05}s`;
                        
                        // Extract string role from object/enum
                        let roleStr = typeof item.role === 'string' ? item.role : Object.keys(item.role)[0] || 'Unknown';
                        
                        div.innerHTML = `
                            <div class="message-header">
                                <span class="role-badge role-${roleStr}">${roleStr}</span>
                                <span>${new Date(item.timestamp * 1000).toLocaleTimeString()}</span>
                            </div>
                            <div class="content">${escapeHtml(item.content)}</div>
                        `;
                        container.appendChild(div);
                    });
                }
            } catch (err) {
                console.error('Fetch error:', err);
            }
        }

        function escapeHtml(unsafe) {
            return (unsafe || '').replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // Poll every 2 seconds
        setInterval(fetchContext, 2000);
        fetchContext();
    </script>
</body>
</html>
    "#)
}
