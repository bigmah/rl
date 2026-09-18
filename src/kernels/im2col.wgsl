// A convolution as a matmul: every place the kernel is laid becomes a row of `cols`, the
// numbers under it in the order a weight's are, (ky, kx, in), so that the convolution is
// cols times the weights' transpose and its gradients are products too.
//
// Pictures are rows from the top with channels last, and a weight is (out, ky, kx, in),
// the layout picture checkpoints already have. The input is a view: each row of it is
// in_stride numbers, with the picture in_off into the row, which is how the first
// convolution reads the picture at the end of an observation.

struct P {
    n: u32, gx: u32,
    in_stride: u32, in_off: u32,
    in_w: u32, in_c: u32,
    out_h: u32, out_w: u32,
    kernel: u32, stride: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> cols: array<f32>;

// A thread for each row of the kernel at each place: what it copies is a run of the input.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let ky = i % p.kernel;
    let place = i / p.kernel;
    let ox = place % p.out_w;
    let oy = (place / p.out_w) % p.out_h;
    let row = place / (p.out_w * p.out_h);
    let run = p.kernel * p.in_c;
    let src = row * p.in_stride + p.in_off + ((oy * p.stride + ky) * p.in_w + ox * p.stride) * p.in_c;
    let dst = i * run;
    for (var j = 0u; j < run; j++) {
        cols[dst + j] = input[src + j];
    }
}
