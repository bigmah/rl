// algo.cu's mingru_scan_backward: the same threads as the forward scan, stepping back
// through time with the gradient of the state in hand.

struct P {
    n: u32, gx: u32,
    horizon: u32,
    hidden: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> combined: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> scan_h: array<f32>;
@group(0) @binding(4) var<storage, read> terminals: array<f32>;
@group(0) @binding(5) var<storage, read> grad_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_combined: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_x: array<f32>;

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
    var dh = 0.0;
    for (var k = T; k > 0u; k--) {
        let t = k - 1u;
        let i = (b * T + t) * H + h;
        let c = (b * T + t) * 3u * H + h;
        let h_prev = scan_h[i];
        let hidden = combined[c];
        let gate = combined[c + H];
        let proj = combined[c + 2u * H];
        let z = sigmoid(gate);
        let h_tilde = select(sigmoid(hidden), hidden + 0.5, hidden >= 0.0);
        let h_t = h_prev + z * (h_tilde - h_prev);
        let s = sigmoid(proj);
        let g = grad_out[i];
        grad_x[i] = g * (1.0 - s);
        grad_combined[c + 2u * H] = g * (h_t - x[i]) * s * (1.0 - s);
        let dh_total = dh + g * s;
        let d_h_tilde = dh_total * z;
        grad_combined[c + H] = dh_total * (h_tilde - h_prev) * z * (1.0 - z);
        grad_combined[c] = select(d_h_tilde * h_tilde * (1.0 - h_tilde), d_h_tilde, hidden >= 0.0);
        // The state before a terminal never reached this step
        dh = select(dh_total * (1.0 - z), 0.0, terminals[b * T + t] != 0.0);
    }
}
