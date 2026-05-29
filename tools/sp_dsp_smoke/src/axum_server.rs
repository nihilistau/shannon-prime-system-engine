//! §3-HX Sprint C — minimal axum HTTP server binding FastRpcSession +
//! DmaBuffer to `POST /v1/dsp/echo` on aarch64-android.
//!
//! Listens on 127.0.0.1:8081 (loopback only per lattice §14.3.1 single-user
//! dev-device rule).  Bring up:
//!
//!     adb push dsp_axum_server /data/local/tmp/
//!     adb shell chmod +x /data/local/tmp/dsp_axum_server
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/dsp_axum_server'
//!
//! Verify from host:
//!
//!     adb forward tcp:8081 tcp:8081
//!     echo -n hello | curl -s --data-binary @- \
//!          -H "Content-Type: application/octet-stream" \
//!          http://127.0.0.1:8081/v1/dsp/echo
//!
//! On non-Android host builds, this binary exits with a message so the
//! `cargo build --target <host>` cycle stays green.

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("dsp_axum_server: host build (target_os != android) — skipped");
    eprintln!("Build with: cargo build --target aarch64-linux-android --release --bin dsp_axum_server");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod server {
    use crate::dsp_rpc::{make_scalars, DmaBuffer, FastRpcSession, RemoteArg, RemoteBuf, SpErr};
    use axum::{
        body::Bytes,
        extract::{DefaultBodyLimit, State},
        http::StatusCode,
        response::IntoResponse,
        routing::post,
        Router,
    };
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    /// Maximum POST body size: 8 MB.  Reduces foot-gun where a runaway client
    /// allocates GB-scale rpcmem buffers.  Bigger payloads → 413.
    const MAX_PAYLOAD: usize = 8 * 1024 * 1024;

    /// Skel URI per qaic-generated sp_echo_URI (sp_echo.h:258).
    const SKEL_URI: &str =
        "file:///libsp_echo_skel.so?sp_echo_skel_handle_invoke&_modver=1.0&_dom=cdsp";

    /// `POST /v1/dsp/echo` body limit override.  Axum's default is 2 MB.
    const ECHO_BODY_LIMIT: usize = MAX_PAYLOAD;

    pub struct AppState {
        pub session: Mutex<FastRpcSession>,
    }

    async fn v1_dsp_echo(
        State(state): State<Arc<AppState>>,
        body: Bytes,
    ) -> Result<(StatusCode, Bytes), (StatusCode, String)> {
        let n = body.len();
        if n == 0 {
            return Err((StatusCode::BAD_REQUEST, "empty body".into()));
        }
        if n > MAX_PAYLOAD {
            return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("body {n} > {MAX_PAYLOAD}")));
        }

        // Spawn-blocking so we don't hold the tokio task on the FastRPC syscall.
        // The session Mutex serializes concurrent requests at the FFI boundary
        // (FastRPC's per-handle thread-safety guarantee is "single thread at a
        // time" — the SDK calculator example uses the same pattern).
        let body = body.to_vec();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, SpErr> {
            let sess = state.session.lock().expect("session mutex poisoned");
            let mut in_buf:  DmaBuffer = sess.alloc_dma(n)?;
            let mut out_buf: DmaBuffer = sess.alloc_dma(n)?;
            in_buf.as_mut_slice().copy_from_slice(&body);
            for b in out_buf.as_mut_slice().iter_mut() { *b = 0; }

            let mut prim_in: [u32; 2] = [n as u32, n as u32];
            let mut args = [
                RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 8 }},
                RemoteArg { buf: RemoteBuf { pv: in_buf.as_mut_ptr() as *mut c_void,  nlen: n }},
                RemoteArg { buf: RemoteBuf { pv: out_buf.as_mut_ptr() as *mut c_void, nlen: n }},
            ];
            sess.invoke(make_scalars(2, 2, 1), &mut args)?;
            Ok(out_buf.as_slice().to_vec())
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")))?;

        match result {
            Ok(out) => Ok((StatusCode::OK, Bytes::from(out))),
            Err(e)  => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e:?}"))),
        }
    }

    pub async fn run() {
        eprintln!("[dsp-axum] opening FastRpcSession (Unsigned PD admission, Path B)...");
        let session = FastRpcSession::new(SKEL_URI).expect("FastRpcSession::new");
        eprintln!("[dsp-axum] session open");

        let state = Arc::new(AppState { session: Mutex::new(session) });
        let app = Router::new()
            .route("/v1/dsp/echo", post(v1_dsp_echo))
            .layer(DefaultBodyLimit::max(ECHO_BODY_LIMIT))
            .with_state(state);

        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8081));
        let listener = tokio::net::TcpListener::bind(addr).await
            .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
        eprintln!("[dsp-axum] listening on http://{addr}/");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                tokio::signal::ctrl_c().await.ok();
                eprintln!("\n[dsp-axum] SIGINT received; shutting down");
            })
            .await
            .expect("axum::serve error");
    }
}

#[cfg(target_os = "android")]
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    server::run().await;
}
