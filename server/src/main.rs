use axum::{
    extract::{Extension, Path},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Model {
    // Simplified representation for demonstration
    parameters: Vec<f32>, // Model parameters (weights, biases, etc.)
    status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ModelStatus {
    Initialized,
    Training,
    Ready,
}

struct ClientInfo {
    model_name: String,
    // Additional client-related information can be added here
}

struct AppState {
    clients: Arc<Mutex<HashMap<String, ClientInfo>>>, // clientIP -> ClientInfo
    models: Arc<Mutex<HashMap<String, Model>>>,       // model_name -> Model
}

#[tokio::main]
async fn main() {
    let app_state = AppState {
        clients: Arc::new(Mutex::new(HashMap::new())),
        models: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/register", post(register_client))
        .route("/init/:model", post(init_model))
        // Add other routes as needed
        .layer(Extension(app_state));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn register_client(
    Json(payload): Json<HashMap<String, String>>, // Expecting {"clientIP": "X.X.X.X", "model": "modelName"}
    Extension(state): Extension<AppState>,
) {
    let client_ip = payload.get("clientIP").unwrap().clone(); // Simplified; real implementation should handle errors
    let model_name = payload.get("model").unwrap().clone();

    let mut clients = state.clients.lock().await;
    clients.insert(client_ip, ClientInfo { model_name });
    println!("Client registered");
}

async fn init_model(
    Path(model_name): Path<String>,
    Extension(state): Extension<AppState>,
) {
    let mut models = state.models.lock().await;
    models.insert(model_name.clone(), Model {
        parameters: vec![], // Initialize with empty or default parameters
        status: ModelStatus::Initialized,
    });
    println!("Model {} initialized", model_name);
}
