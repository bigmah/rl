// C = A * B, or C += that: A is m by k and B is k by n, each a view of its buffer by an
// offset and the strides of a row and a column, so that a transpose, or a block of the
// flat weights, is a choice of strides. C's rows are c_rs apart.
//
// A thread computes a 4 by 4 block of C: each step along k it reads four numbers of A and
// four of B and makes sixteen products of them, which is what keeps a GPU's arithmetic
// busier than its memory.
//
// A GPU thread is slow alone, a step of k taking it a tenth of a microsecond, and what a
// GPU has is threads. So a long k, or the k of a product small enough to leave the GPU
// idle, is cut into chunks, a layer of the dispatch for each: a layer writes its partial
// sums to its own m by n, and sum_layers.wgsl adds them into C. Matrices of one shape
// multiply together as a batch, which is layers of the dispatch again: each batch's A, B
// and C are a_bs, b_bs and c_zs on from the last's.

struct P {
    m: u32, n: u32, k: u32,
    a_off: u32, a_rs: u32, a_cs: u32,
    b_off: u32, b_rs: u32, b_cs: u32,
    c_off: u32, c_rs: u32,
    accumulate: u32,
    chunk: u32,
    layers: u32,  // chunks of k in each batch
    a_bs: u32, b_bs: u32,
    c_zs: u32,    // from one layer of the dispatch's output to the next
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row0 = gid.y * 4u;
    let col0 = gid.x * 4u;
    if (row0 >= p.m || col0 >= p.n) {
        return;
    }
    // Rows and columns past the edge read the last one again, and are not written
    let rmax = p.m - 1u;
    let cmax = p.n - 1u;
    let batch = gid.z / p.layers;
    let ra = p.a_off + batch * p.a_bs + vec4<u32>(row0, min(row0 + 1u, rmax), min(row0 + 2u, rmax), min(row0 + 3u, rmax)) * p.a_rs;
    let cb = p.b_off + batch * p.b_bs + vec4<u32>(col0, min(col0 + 1u, cmax), min(col0 + 2u, cmax), min(col0 + 3u, cmax)) * p.b_cs;
    let k_lo = (gid.z % p.layers) * p.chunk;
    let k_hi = min(p.k, k_lo + p.chunk);
    var s0 = vec4<f32>(0.0);
    var s1 = vec4<f32>(0.0);
    var s2 = vec4<f32>(0.0);
    var s3 = vec4<f32>(0.0);
    for (var k = k_lo; k < k_hi; k++) {
        let ka = k * p.a_cs;
        let kb = k * p.b_rs;
        let bv = vec4<f32>(b[cb.x + kb], b[cb.y + kb], b[cb.z + kb], b[cb.w + kb]);
        s0 += a[ra.x + ka] * bv;
        s1 += a[ra.y + ka] * bv;
        s2 += a[ra.z + ka] * bv;
        s3 += a[ra.w + ka] * bv;
    }
    let sums = array<vec4<f32>, 4>(s0, s1, s2, s3);
    let layer = p.c_off + gid.z * p.c_zs;
    for (var i = 0u; i < 4u; i++) {
        let row = row0 + i;
        if (row >= p.m) {
            break;
        }
        for (var j = 0u; j < 4u; j++) {
            let col = col0 + j;
            if (col >= p.n) {
                break;
            }
            let at = layer + row * p.c_rs + col;
            if (p.accumulate != 0u) {
                c[at] += sums[i][j];
            } else {
                c[at] = sums[i][j];
            }
        }
    }
}
