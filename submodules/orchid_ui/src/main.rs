use axum::{body::Body, extract::{Request, State}, http::{header, StatusCode}, middleware::{self, Next}, response::Response, routing::get, Json, Router};
use serde_json::{json, Value};
use std::{net::SocketAddr, sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant}};
use tokio::sync::{oneshot, Mutex};
use tower_http::services::ServeDir;

struct Prompt { id: u64, text: String, choices: Vec<Value>, responder: oneshot::Sender<String> }
struct AppState { pending: Mutex<Option<Prompt>>, next_id: AtomicU64, started: Instant, timeout: Duration }
impl AppState {
    fn new(timeout: Duration) -> Self { Self { pending: Mutex::new(None), next_id: AtomicU64::new(1), started: Instant::now(), timeout } }
    #[cfg(test)]
    async fn install(&self, text: String) -> (u64, oneshot::Receiver<String>) {
        self.install_with_choices(text, Vec::new()).await
    }
    async fn install_with_choices(&self, text: String, choices: Vec<Value>) -> (u64, oneshot::Receiver<String>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        *self.pending.lock().await = Some(Prompt { id, text, choices, responder: tx });
        (id, rx)
    }
    async fn clear_owned(&self, id: u64) {
        let mut pending = self.pending.lock().await;
        if pending.as_ref().map(|p| p.id) == Some(id) { *pending = None; }
    }
    async fn answer(&self, id: Option<u64>, answer: String) -> bool {
        let mut pending = self.pending.lock().await;
        if pending.as_ref().is_none() || id.is_some_and(|id| pending.as_ref().unwrap().id != id) { return false; }
        pending.take().is_some_and(|prompt| prompt.responder.send(answer).is_ok())
    }
}

async fn request_log(request: Request<Body>, next: Next) -> Response {
    let method = request.method().to_string(); let path = request.uri().path().to_string(); let began = Instant::now();
    let mut response = next.run(request).await;
    if path != "/healthz" { response.headers_mut().insert(header::CACHE_CONTROL, "no-cache".parse().unwrap()); }
    if path == "/healthz" || path == "/" || path == "/index.html" || response.status().is_client_error() || response.status().is_server_error() {
        println!("{}", json!({"http_method":method,"path":path,"status":response.status().as_u16(),"elapsed_ms":began.elapsed().as_millis()}));
    }
    response
}
async fn health(State(state): State<Arc<AppState>>) -> ([(header::HeaderName, &'static str);1], Json<Value>) {
    ([(header::CACHE_CONTROL,"no-store")], Json(json!({"ok":true,"uptime_s":state.started.elapsed().as_secs()})))
}
#[tokio::main]
async fn main() {
    let timeout = std::env::var("ORCHID_UI_FOLLOWUP_TIMEOUT_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let port = std::env::var("ORCHID_UI_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8000);
    let state = Arc::new(AppState::new(Duration::from_secs(timeout)));
    let app = Router::new().route("/healthz",get(health))
        .route("/api/followup",get(handle_pending).post(handle_followup))
        .nest_service("/",ServeDir::new("static")).layer(middleware::from_fn(request_log)).with_state(state);
    let addr=SocketAddr::from(([0,0,0,0],port));
    let listener=tokio::net::TcpListener::bind(addr).await.expect("dashboard HTTP bind failed");
    println!("Orchid UI web server starting on http://{addr}");
    axum::serve(listener,app).await.expect("dashboard HTTP server failed");
}
async fn handle_followup(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> (StatusCode,Json<Value>) {
    let prompt_id=body.get("prompt_id").and_then(Value::as_u64);
    if let Some(answer)=body.get("response").and_then(Value::as_str) {
        let ok=state.answer(prompt_id,answer.to_string()).await;
        return (if ok {StatusCode::OK}else{StatusCode::CONFLICT},Json(json!({"ok":ok})));
    }
    if body.get("cancel").and_then(Value::as_bool)==Some(true) {
        let ok=state.answer(prompt_id,String::new()).await;
        return (if ok || prompt_id.is_none(){StatusCode::OK}else{StatusCode::CONFLICT},Json(json!({"ok":ok,"cancelled":ok})));
    }
    if let Some(prompt)=body.get("prompt").and_then(Value::as_str) {
        let choices = match body.get("choices") {
            None => Vec::new(),
            Some(Value::Array(values)) if values.iter().all(|v| v.get("value").and_then(Value::as_str).is_some() && v.get("label").and_then(Value::as_str).is_some()) => values.clone(),
            _ => return (StatusCode::BAD_REQUEST, Json(json!({"error":"choices must contain value/label strings"}))),
        };
        let timeout = match body.get("timeout_s") {
            None => state.timeout,
            Some(value) => match value.as_f64() {
                Some(seconds) if seconds.is_finite() && seconds > 0. && seconds <= 86400. => Duration::from_secs_f64(seconds),
                _ => return (StatusCode::BAD_REQUEST,Json(json!({"error":"invalid timeout_s"}))),
            },
        };
        let (id,rx)=state.install_with_choices(prompt.to_string(),choices).await;
        let result=tokio::time::timeout(timeout,rx).await;
        state.clear_owned(id).await;
        return match result {
            Ok(Ok(response))=>(StatusCode::OK,Json(json!({"response":response}))),
            _=>(StatusCode::REQUEST_TIMEOUT,Json(json!({"error":"timeout","prompt_id":id}))),
        };
    }
    (StatusCode::BAD_REQUEST,Json(json!({"error":"expected prompt, response or cancel"})))
}
async fn handle_pending(State(state):State<Arc<AppState>>) -> Json<Value> {
    let pending=state.pending.lock().await;
    Json(match pending.as_ref(){Some(p)=>json!({"prompt":p.text,"prompt_id":p.id,"choices":p.choices}),None=>json!({"prompt":null,"prompt_id":null,"choices":[]})})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test] async fn choices_survive_pending_prompt_and_ack() {
        let state=AppState::new(Duration::from_secs(1));
        let choices=vec![json!({"value":"table_1","label":"Table — kitchen"})];
        let (id,rx)=state.install_with_choices("Where?".into(),choices.clone()).await;
        assert_eq!(state.pending.lock().await.as_ref().unwrap().choices,choices);
        assert!(state.answer(Some(id),"table_1".into()).await);
        assert_eq!(rx.await.unwrap(),"table_1");
    }
    #[tokio::test] async fn answer_and_stale_reply() {
        let state=AppState::new(Duration::from_secs(1)); let (id,rx)=state.install("question".into()).await;
        assert!(!state.answer(Some(id+1),"stale".into()).await);
        assert!(state.answer(Some(id),"yes".into()).await); assert_eq!(rx.await.unwrap(),"yes");
        assert!(!state.answer(Some(id),"again".into()).await);
    }
    #[tokio::test] async fn old_cleanup_does_not_remove_new_question() {
        let state=AppState::new(Duration::from_secs(1)); let (old,_)=state.install("old".into()).await;
        let (new,rx)=state.install("new".into()).await; state.clear_owned(old).await;
        assert!(state.answer(Some(new),"new answer".into()).await); assert_eq!(rx.await.unwrap(),"new answer");
    }
    #[tokio::test] async fn legacy_answer_and_cancel() {
        let state=AppState::new(Duration::from_secs(1)); let (_,rx)=state.install("question".into()).await;
        assert!(state.answer(None,String::new()).await); assert_eq!(rx.await.unwrap(),"");
    }
    #[tokio::test] async fn concurrent_replacement_and_expiry() {
        let state=Arc::new(AppState::new(Duration::from_secs(1))); let (old,_)=state.install("old".into()).await;
        let a=state.clone(); let b=state.clone();
        let (_, (new,rx))=tokio::join!(a.clear_owned(old),b.install("new".into()));
        assert!(state.answer(Some(new),"ok".into()).await); assert_eq!(rx.await.unwrap(),"ok");
    }
}
