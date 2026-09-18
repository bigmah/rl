// The way back from im2col.wgsl: the gradient of `cols` gathered into the gradient of the
// picture it was cut from, a thread for each number of the picture, which sums over every
// place the kernel had that number under it.

struct P {
    n: u32, gx: u32,
    in_h: u32, in_w: u32, in_c: u32,
    out_h: u32, out_w: u32,
    kernel: u32, stride: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad_cols: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let c = i % p.in_c;
    let ix = (i / p.in_c) % p.in_w;
    let iy = (i / (p.in_c * p.in_w)) % p.in_h;
    let row = i / (p.in_c * p.in_w * p.in_h);
    let under_kernel = p.kernel * p.kernel * p.in_c;
    var sum = 0.0;
    for (var ky = 0u; ky < p.kernel; ky++) {
        if (iy < ky || (iy - ky) % p.stride != 0u) {
            continue;
        }
        let oy = (iy - ky) / p.stride;
        if (oy >= p.out_h) {
            continue;
        }
        for (var kx = 0u; kx < p.kernel; kx++) {
            if (ix < kx || (ix - kx) % p.stride != 0u) {
                continue;
            }
            let ox = (ix - kx) / p.stride;
            if (ox >= p.out_w) {
                continue;
            }
            let place = (row * p.out_h + oy) * p.out_w + ox;
            sum += grad_cols[place * under_kernel + (ky * p.kernel + kx) * p.in_c + c];
        }
    }
    grad_in[i] = sum;
}
