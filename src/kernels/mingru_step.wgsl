// One step of one MinGRU layer for every agent of a rollout: algo.cu's mingru_gate.
// combined is the layer's matrix applied to x, as hidden, gate and highway; the state is
// cleared first for an agent whose episode just ended.

struct P {
    n: u32, gx: u32,
    hidden: u32,
    state_off: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> combined: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> terminals: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

fn sigmoid(v: f32) -> f32 {
    let z = exp(-abs(v));
    return select(z / (1.0 + z), 1.0 / (1.0 + z), v >= 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let H = p.hidden;
    let row = i / H;
    let h = i % H;
    let c = row * 3u * H + h;
    let hidden = combined[c];
    let gate = combined[c + H];
    let proj = combined[c + 2u * H];

    var prev = state[p.state_off + i];
    if (terminals[row] != 0.0) {
        prev = 0.0;
    }
    let z = sigmoid(gate);
    let h_tilde = select(sigmoid(hidden), hidden + 0.5, hidden >= 0.0);
    let next = prev + z * (h_tilde - prev);
    state[p.state_off + i] = next;
    let s = sigmoid(proj);
    out[i] = s * next + (1.0 - s) * x[i];
}
