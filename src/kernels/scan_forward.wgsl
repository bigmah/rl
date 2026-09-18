// algo.cu's mingru_scan_forward: one MinGRU layer over whole segments. A thread for each
// unit of each segment steps through time, clearing the state where a terminal says a new
// episode starts. scan_h keeps the state each step started from, for the backward pass.

struct P {
    n: u32, gx: u32,
    horizon: u32,
    hidden: u32,
    state_off: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> combined: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> state: array<f32>;
@group(0) @binding(4) var<storage, read> terminals: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read_write> scan_h: array<f32>;

fn sigmoid(v: f32) -> f32 {
    let z = exp(-abs(v));
    return select(z / (1.0 + z), 1.0 / (1.0 + z), v >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x + gid.y * p.gx;
    if (thread >= p.n) {
        return;
    }
    let T = p.horizon;
    let H = p.hidden;
    let b = thread / H;
    let h = thread % H;
    var state_h = state[p.state_off + thread];
    for (var t = 0u; t < T; t++) {
        let i = (b * T + t) * H + h;
        let c = (b * T + t) * 3u * H + h;
        if (terminals[b * T + t] != 0.0) {
            state_h = 0.0;
        }
        scan_h[i] = state_h;
        let hidden = combined[c];
        let gate = combined[c + H];
        let proj = combined[c + 2u * H];
        let z = sigmoid(gate);
        let h_tilde = select(sigmoid(hidden), hidden + 0.5, hidden >= 0.0);
        state_h = state_h + z * (h_tilde - state_h);
        let s = sigmoid(proj);
        out[i] = s * state_h + (1.0 - s) * x[i];
    }
}
