// The end of a matmul in layers: each batch's layers of partial sums, added into its C.

struct P {
    n: u32, gx: u32,
    size: u32,   // m * n, of one matrix
    cols: u32,
    layers: u32,
    c_off: u32, c_rs: u32, c_bs: u32,
    accumulate: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let batch = i / p.size;
    let e = i % p.size;
    var sum = 0.0;
    for (var layer = 0u; layer < p.layers; layer++) {
        sum += partials[(batch * p.layers + layer) * p.size + e];
    }
    let at = p.c_off + batch * p.c_bs + (e / p.cols) * p.c_rs + e % p.cols;
    if (p.accumulate != 0u) {
        c[at] += sum;
    } else {
        c[at] = sum;
    }
}
