// The first half of algo.cu's Muon, over every weight at once: the gradient clipped to
// max_grad_norm by its global norm, Nesterov momentum on the clipped gradient, and the
// update that Newton-Schulz then orthogonalizes matrix by matrix.

struct P {
    n: u32, gx: u32,
    momentum: f32,
    max_grad_norm: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read> grad_norm_sq: array<f32>;
@group(0) @binding(3) var<storage, read_write> momentum: array<f32>;
@group(0) @binding(4) var<storage, read_write> update: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let clip = min(p.max_grad_norm / (sqrt(grad_norm_sq[0]) + 1e-6), 1.0);
    let g = grads[i] * clip;
    let m = p.momentum * momentum[i] + g;
    momentum[i] = m;
    update[i] = g + p.momentum * m;
}
