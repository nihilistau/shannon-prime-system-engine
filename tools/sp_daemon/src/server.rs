use std::sync::Arc;

use axum::{routing::get, Router};

use crate::{routes::v1_metrics, state::AppState};

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/metrics", get(v1_metrics))
        .with_state(state)
}
