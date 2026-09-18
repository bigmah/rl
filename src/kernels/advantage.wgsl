// algo.cu's puff_advantage, a thread for each segment: GAE, with V-trace's clipped
// importance weights on the TD error and on the trace when vtrace is on. The reward and
// terminal stored with a step are the ones the step before it left, so step t looks at
// t + 1 for both. The last step's advantage is 0.

struct P {
    n: u32, gx: u32,
    horizon: u32,
    vtrace: u32,
    gamma: f32,
    lambda: f32,
    rho_clip: f32,
    c_clip: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> rewards: array<f32>;
@group(0) @binding(3) var<storage, read> terminals: array<f32>;
@group(0) @binding(4) var<storage, read> ratio: array<f32>;
@group(0) @binding(5) var<storage, read_write> advantages: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x + gid.y * p.gx;
    if (row >= p.n) {
        return;
    }
    let T = p.horizon;
    let off = row * T;
    advantages[off + T - 1u] = 0.0;
    var last = 0.0;
    for (var k = T - 1u; k > 0u; k--) {
        let t = k - 1u;
        let nonterminal = 1.0 - terminals[off + t + 1u];
        var importance = 1.0;
        if (p.vtrace != 0u) {
            importance = ratio[off + t];
        }
        let rho = min(importance, p.rho_clip);
        let c = min(importance, p.c_clip);
        let delta = rho * (rewards[off + t + 1u] + p.gamma * values[off + t + 1u] * nonterminal - values[off + t]);
        last = delta + p.gamma * p.lambda * c * nonterminal * last;
        advantages[off + t] = last;
    }
}
