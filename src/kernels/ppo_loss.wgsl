// algo.cu's ppo_loss_compute: the clipped policy loss, the clipped value loss and the
// entropy bonus of each step, and in the same walk their gradient with respect to the
// step's logits and value, which is where the backward pass starts. Each step also leaves
// its share of the eight statistics pufferl.cu logs, to be summed afterwards.
//
// Advantages come from this minibatch's own values before the update, and nothing flows
// back through them. With norm_adv the policy term sees them centered and scaled to unit
// variance, as PufferLib 4.0 did; the returns the value learns are from the raw ones.

//HEADS

struct P {
    n: u32, gx: u32,
    norm_adv: u32,
    clip_coef: f32,
    vf_clip_coef: f32,
    vf_coef: f32,
}

struct Hyper {
    learning_rate: f32,
    ent_coef: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> hyper: Hyper;
@group(0) @binding(2) var<storage, read> dec: array<f32>;
@group(0) @binding(3) var<storage, read> actions: array<u32>;
@group(0) @binding(4) var<storage, read> logratio: array<f32>;
@group(0) @binding(5) var<storage, read> old_values: array<f32>;
@group(0) @binding(6) var<storage, read> advantages: array<f32>;
// The sum of the advantages, and of their squared distances from their mean
@group(0) @binding(7) var<storage, read> adv_sums: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_dec: array<f32>;
@group(0) @binding(9) var<storage, read_write> partials: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.gx;
    if (i >= p.n) {
        return;
    }
    let inv_n = 1.0 / f32(p.n);
    let base = i * (NUM_LOGITS + 1u);

    let raw_adv = advantages[i];
    var adv = raw_adv;
    if (p.norm_adv != 0u) {
        let mean = adv_sums[0] * inv_n;
        let deviation = sqrt(adv_sums[1] * inv_n);
        adv = (raw_adv - mean) / (deviation + 1e-8);
    }

    // Value: 0.5 * max((v - R)^2, (v_clipped - R)^2). Inside the clip both are the same
    // number, and outside it the clipped one is a constant.
    let value = dec[base + NUM_LOGITS];
    let ret = value + raw_adv;
    let old_value = old_values[i];
    let v_error = value - old_value;
    let v_clipped = old_value + clamp(v_error, -p.vf_clip_coef, p.vf_clip_coef);
    let v_unclipped_loss = (value - ret) * (value - ret);
    let v_clipped_loss = (v_clipped - ret) * (v_clipped - ret);
    let v_loss = 0.5 * max(v_unclipped_loss, v_clipped_loss);
    var d_value = value - ret;
    if (abs(v_error) > p.vf_clip_coef && v_clipped_loss > v_unclipped_loss) {
        d_value = 0.0;
    }
    grad_dec[base + NUM_LOGITS] = inv_n * p.vf_coef * d_value;

    // Policy: max(-A * ratio, -A * clip(ratio))
    let lr = logratio[i];
    let ratio = exp(lr);
    let clip_lo = 1.0 - p.clip_coef;
    let clip_hi = 1.0 + p.clip_coef;
    let ratio_clipped = clamp(ratio, clip_lo, clip_hi);
    let pg1 = -adv * ratio;
    let pg2 = -adv * ratio_clipped;
    let pg_loss = max(pg1, pg2);
    var d_ratio = -adv * inv_n;
    if (pg2 > pg1 && (ratio <= clip_lo || ratio >= clip_hi)) {
        d_ratio = 0.0;
    }
    let d_logprob = d_ratio * ratio;
    let d_entropy = -hyper.ent_coef * inv_n;

    var entropy = 0.0;
    var offset = 0u;
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
        let lse = top + log(total);
        var head_entropy = 0.0;
        for (var j = 0u; j < A; j++) {
            let logp = dec[base + offset + j] - lse;
            head_entropy -= exp(logp) * logp;
        }
        entropy += head_entropy;
        let taken = actions[i * NUM_HEADS + h];
        for (var j = 0u; j < A; j++) {
            let logp = dec[base + offset + j] - lse;
            let prob = exp(logp);
            let onehot = select(0.0, 1.0, j == taken);
            grad_dec[base + offset + j] = (onehot - prob) * d_logprob + d_entropy * prob * (-head_entropy - logp);
        }
        offset += A;
    }

    let at = i * 8u;
    partials[at] = pg_loss * inv_n;
    partials[at + 1u] = v_loss * inv_n;
    partials[at + 2u] = entropy * inv_n;
    partials[at + 3u] = (pg_loss + p.vf_coef * v_loss - hyper.ent_coef * entropy) * inv_n;
    partials[at + 4u] = -lr * inv_n;
    partials[at + 5u] = ((ratio - 1.0) - lr) * inv_n;
    partials[at + 6u] = select(0.0, inv_n, abs(ratio - 1.0) > p.clip_coef);
    partials[at + 7u] = ratio * inv_n;
}
