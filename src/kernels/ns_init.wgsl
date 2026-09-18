// Newton-Schulz starts from each matrix's update over its Frobenius norm, laid out wide: a
// matrix with more rows than columns goes in transposed, so that x * x^T is the small
// square. The matrices of a batch are one shape, `stride` apart in the flat update.

struct P {
    n: u32, gx: u32,
    off: u32,
    stride: u32,
    size: u32,       // of one matrix
    cols: u32,       // of the weight matrix
    wide_cols: u32,  // of x
    transpose: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@group(0) @binding(2) var<storage, read> norm_sq: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let batch = i / p.size;
    let e = i % p.size;
    let r = e / p.wide_cols;
    let c = e % p.wide_cols;
    let src = select(r * p.cols + c, c * p.cols + r, p.transpose != 0u);
    x[i] = update[p.off + batch * p.stride + src] / max(sqrt(norm_sq[batch]), 1e-7);
}
