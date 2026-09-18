// The end of a Muon step for a batch of matrices of one shape: each one's orthogonalized
// update, turned back the way the weights lie and scaled by sqrt(max(1, rows / cols)),
// taken off the weights at the learning rate, with weight decay if there is any.

struct P {
    n: u32, gx: u32,
    off: u32,
    stride: u32,
    size: u32,
    cols: u32,
    wide_cols: u32,
    transpose: u32,
    scale: f32,
    weight_decay: f32,
}

struct Hyper {
    learning_rate: f32,
    ent_coef: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> hyper: Hyper;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let batch = i / p.size;
    let e = i % p.size;
    let r = e / p.cols;
    let c = e % p.cols;
    let src = batch * p.size + select(r * p.wide_cols + c, c * p.wide_cols + r, p.transpose != 0u);
    let at = p.off + batch * p.stride + e;
    let lr = hyper.learning_rate;
    weights[at] = weights[at] * (1.0 - lr * p.weight_decay) - lr * p.scale * x[src];
}
