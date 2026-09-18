// algo.cu's cache_imp_and_v: after the decoder and before the advantages, one walk over
// each step's logits for the log-probability of the action the rollout took, and with it
// the importance ratio against the policy that took it; and the value, pulled out of the
// decoder's last column.

//HEADS

struct P {
    n: u32, gx: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dec: array<f32>;
@group(0) @binding(2) var<storage, read> actions: array<u32>;
@group(0) @binding(3) var<storage, read> old_logprobs: array<f32>;
@group(0) @binding(4) var<storage, read_write> logratio: array<f32>;
@group(0) @binding(5) var<storage, read_write> ratio: array<f32>;
@group(0) @binding(6) var<storage, read_write> values: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let base = i * (NUM_LOGITS + 1u);
    var offset = 0u;
    var logprob = 0.0;
    for (var h = 0u; h < NUM_HEADS; h++) {
        let A = ACT_SIZES[h];
        var top = dec[base + offset];
        for (var j = 1u; j < A; j++) {
            top = max(top, dec[base + offset + j]);
        }
        var total = 0.0;
        for (var j = 0u; j < A; j++) {
            total += exp(dec[base + offset + j] - top);
        }
        logprob += dec[base + offset + actions[i * NUM_HEADS + h]] - (top + log(total));
        offset += A;
    }
    let lr = logprob - old_logprobs[i];
    logratio[i] = lr;
    ratio[i] = exp(lr);
    values[i] = dec[base + NUM_LOGITS];
}
