//! Training (src/algo.cu): PPO as PufferLib 5.0 has it, and Muon.
//!
//! A minibatch is whole segments, a run of agents' horizons cut from the rollout. Its
//! forward pass, its loss, its backward pass and its optimizer step are one program, built
//! when the learner is, and run once for every minibatch with nothing read back: the loss
//! statistics add up on the GPU and come back once an epoch.

use std::rc::Rc;

use crate::actor::Rollout;
use crate::gpu::{Buf, Gpu, Params, Program};
use crate::model::Model;
use crate::ops::{self, matmul_batched, Out, Reduce, Sum, View};

const SCAN_FORWARD: &str = include_str!("kernels/scan_forward.wgsl");
const SCAN_BACKWARD: &str = include_str!("kernels/scan_backward.wgsl");
const PPO_PRE: &str = include_str!("kernels/ppo_pre.wgsl");
const ADVANTAGE: &str = include_str!("kernels/advantage.wgsl");
const PPO_LOSS: &str = include_str!("kernels/ppo_loss.wgsl");
const MUON_MOMENTUM: &str = include_str!("kernels/muon_momentum.wgsl");
const NS_INIT: &str = include_str!("kernels/ns_init.wgsl");
const NS_POLY: &str = include_str!("kernels/ns_poly.wgsl");
const MUON_APPLY: &str = include_str!("kernels/muon_apply.wgsl");

/// The statistics pufferl.cu logs, in the order the loss kernel leaves them.
pub const LOSS_KEYS: [&str; 8] = ["loss/policy", "loss/value", "loss/entropy", "loss/total",
    "loss/old_kl", "loss/kl", "loss/clipfrac", "importance"];

const NS_COEFS: [(f32, f32, f32); 5] = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
];

/// What of [train] stays the same all run.
#[derive(Clone, Debug)]
pub struct Hyper {
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_coef: f32,
    pub vf_coef: f32,
    pub vf_clip_coef: f32,
    pub max_grad_norm: f32,
    pub momentum: f32,
    /// PufferLib has none, so 0 is its trainer
    pub weight_decay: f32,
    pub vtrace: bool,
    pub vtrace_rho_clip: f32,
    pub vtrace_c_clip: f32,
    /// Not in 5.0: each minibatch's advantages centered and scaled, as 4.0 did
    pub norm_adv: bool,
}

/// A minibatch, cut from the rollout.
pub struct Minibatch {
    pub obs: Buf,
    pub actions: Buf,
    pub logprobs: Buf,
    pub values: Buf,
    pub rewards: Buf,
    pub terminals: Buf,
    pub state: Buf,
}

pub struct Learner {
    gpu: Rc<Gpu>,
    pub segments: usize,
    pub horizon: usize,
    agents: usize,
    hyper_uniform: wgpu::Buffer,
    pub minibatch: Minibatch,
    pub grads: Buf,
    pub momentum: Buf,
    stats: Buf,
    stats_staging: Buf,
    /// One for every place in the rollout a minibatch starts
    cuts: Vec<Program>,
    /// Everything but the optimizer, for looking at gradients
    pub forward_backward: Program,
    pub optimizer: Program,
    clear_stats: Program,
    read_stats: Program,
}

impl Learner {
    pub fn new(model: &Model, rollout: &Rollout, segments: usize, hyper: &Hyper) -> Self {
        let gpu = model.gpu.clone();
        let s = &model.shapes;
        let (hidden, layers, heads, dec_cols) = (s.hidden, s.layers, s.num_heads(), s.dec_cols());
        let (agents, horizon) = (rollout.agents, rollout.horizon);
        assert!(segments > 0 && agents % segments == 0, "agents must divide into minibatches of whole segments");
        let rows = segments * horizon;

        let minibatch = Minibatch {
            obs: gpu.storage("mb.obs", rows * s.obs_size),
            actions: gpu.storage("mb.actions", rows * heads),
            logprobs: gpu.storage("mb.logprobs", rows),
            values: gpu.storage("mb.values", rows),
            rewards: gpu.storage("mb.rewards", rows),
            terminals: gpu.storage("mb.terminals", rows),
            state: gpu.storage("mb.state", layers * segments * hidden),
        };
        let mb = &minibatch;

        // Where a minibatch that starts at agent `lo` is in the rollout
        let cuts = (0..agents / segments).map(|i| {
            let lo = i * segments;
            let mut cut = Program::new();
            let at = lo * horizon;
            cut.copy(&rollout.obs, at * s.obs_size, &mb.obs, 0, rows * s.obs_size);
            cut.copy(&rollout.actions, at * heads, &mb.actions, 0, rows * heads);
            cut.copy(&rollout.logprobs, at, &mb.logprobs, 0, rows);
            cut.copy(&rollout.values, at, &mb.values, 0, rows);
            cut.copy(&rollout.rewards, at, &mb.rewards, 0, rows);
            cut.copy(&rollout.terminals, at, &mb.terminals, 0, rows);
            for layer in 0..layers {
                cut.copy(&rollout.initial_state, (layer * agents + lo) * hidden,
                    &mb.state, layer * segments * hidden, segments * hidden);
            }
            cut
        }).collect();

        let hyper_uniform = gpu.uniform("hyper", &[0u8; 16]);
        let grads = model.flat("grads");
        let momentum = model.flat("momentum");
        let stats = gpu.storage("loss.stats", LOSS_KEYS.len());
        let stats_staging = gpu.staging("loss.stats.staging", LOSS_KEYS.len());
        let heads_wgsl = s.heads_wgsl();

        // Forward, keeping what the backward pass needs: every layer's input, what its
        // matrix made of it, and the state each step of its scan started from
        let acts = model.encoder_acts(rows, true);
        let x: Vec<Buf> = (0..=layers).map(|i| gpu.storage(&format!("train.x.{i}"), rows * hidden)).collect();
        let combined: Vec<Buf> = (0..layers).map(|i| gpu.storage(&format!("train.combined.{i}"), rows * 3 * hidden)).collect();
        let scan_h: Vec<Buf> = (0..layers).map(|i| gpu.storage(&format!("train.scan_h.{i}"), rows * hidden)).collect();
        let dec = gpu.storage("train.dec", rows * dec_cols);

        let mut fb = Program::new();
        fb.extend(model.encoder_forward(rows, &mb.obs, &acts, &x[0]));
        for layer in 0..layers {
            fb.extend(model.linear(rows, &x[layer], model.layer(layer), &combined[layer]));
            let rest = Params::new().u(horizon).u(hidden).u(layer * segments * hidden);
            fb.push(gpu.op_flat("scan_forward", SCAN_FORWARD, segments * hidden, rest,
                &[&combined[layer].buffer, &x[layer].buffer, &mb.state.buffer, &mb.terminals.buffer,
                  &x[layer + 1].buffer, &scan_h[layer].buffer]));
        }
        fb.extend(model.linear(rows, &x[layers], model.decoder(), &dec));

        // The loss, and the gradient it starts the backward pass with
        let logratio = gpu.storage("loss.logratio", rows);
        let ratio = gpu.storage("loss.ratio", rows);
        let live_values = gpu.storage("loss.values", rows);
        let advantages = gpu.storage("loss.advantages", rows);
        let adv_sums = gpu.storage("loss.adv_sums", 2);
        let partials = gpu.storage("loss.partials", rows * LOSS_KEYS.len());
        let scratch = gpu.storage("reduce.scratch", ops::scratch_len(LOSS_KEYS.len()));
        let grad_dec = gpu.storage("train.grad_dec", rows * dec_cols);

        fb.push(gpu.op_flat("ppo_pre", &PPO_PRE.replace("//HEADS", &heads_wgsl), rows, Params::new(),
            &[&dec.buffer, &mb.actions.buffer, &mb.logprobs.buffer, &logratio.buffer, &ratio.buffer, &live_values.buffer]));
        let rest = Params::new().u(horizon).u(hyper.vtrace as usize)
            .f(hyper.gamma).f(hyper.gae_lambda).f(hyper.vtrace_rho_clip).f(hyper.vtrace_c_clip);
        fb.push(gpu.op_flat("advantage", ADVANTAGE, segments, rest,
            &[&live_values.buffer, &mb.rewards.buffer, &mb.terminals.buffer, &ratio.buffer, &advantages.buffer]));
        if hyper.norm_adv {
            fb.extend(ops::mean_and_deviation(&gpu, &advantages, rows, &adv_sums, &scratch));
        }
        let rest = Params::new().u(hyper.norm_adv as usize).f(hyper.clip_coef).f(hyper.vf_clip_coef).f(hyper.vf_coef);
        fb.push(gpu.op_flat("ppo_loss", &PPO_LOSS.replace("//HEADS", &heads_wgsl), rows, rest,
            &[&hyper_uniform, &dec.buffer, &mb.actions.buffer, &logratio.buffer, &mb.values.buffer,
              &advantages.buffer, &adv_sums.buffer, &grad_dec.buffer, &partials.buffer]));
        fb.extend(ops::reduce(&gpu, Reduce { input: &partials, in_off: 0, n_in: rows * LOSS_KEYS.len(),
            columns: LOSS_KEYS.len(), sum: Sum::Values, output: &stats, out_off: 0, accumulate: true, scratch: &scratch }));

        // Backward, from the decoder down. A layer's input gets gradient two ways, through
        // the layer's matrix and straight across its highway, so the second is added to
        // the first.
        let mut grad_x = [gpu.storage("train.grad_x.0", rows * hidden), gpu.storage("train.grad_x.1", rows * hidden)];
        let grad_combined = gpu.storage("train.grad_combined", rows * 3 * hidden);
        fb.extend(model.linear_grad_weight(rows, &grad_dec, &x[layers], model.decoder(), &grads));
        fb.extend(model.linear_grad_input(rows, &grad_dec, model.decoder(), &grad_x[0], false));
        for layer in (0..layers).rev() {
            let w = model.layer(layer);
            fb.push(gpu.op_flat("scan_backward", SCAN_BACKWARD, segments * hidden, Params::new().u(horizon).u(hidden),
                &[&combined[layer].buffer, &x[layer].buffer, &scan_h[layer].buffer, &mb.terminals.buffer,
                  &grad_x[0].buffer, &grad_combined.buffer, &grad_x[1].buffer]));
            fb.extend(model.linear_grad_weight(rows, &grad_combined, &x[layer], w, &grads));
            fb.extend(model.linear_grad_input(rows, &grad_combined, w, &grad_x[1], true));
            grad_x.swap(0, 1);
        }
        fb.extend(model.encoder_backward(rows, &mb.obs, &acts, &grad_x[0], &grads));

        let optimizer = Self::muon(model, &grads, &momentum, &hyper_uniform, hyper);

        let mut clear_stats = Program::new();
        clear_stats.clear(&stats);
        let mut read_stats = Program::new();
        read_stats.copy(&stats, 0, &stats_staging, 0, LOSS_KEYS.len());

        Self { gpu, segments, horizon, agents, hyper_uniform, minibatch, grads, momentum, stats, stats_staging,
            cuts, forward_backward: fb, optimizer, clear_stats, read_stats }
    }

    /// algo.cu's Muon: Nesterov momentum on the clipped gradient, orthogonalized matrix by
    /// matrix with five steps of Newton-Schulz. Matrices of one shape that lie together in
    /// the weights, as the MinGRU's layers do, are orthogonalized together as a batch.
    fn muon(model: &Model, grads: &Buf, momentum: &Buf, hyper_uniform: &wgpu::Buffer, hyper: &Hyper) -> Program {
        let gpu = &model.gpu;
        let update = model.flat("muon.update");
        let grad_norm_sq = gpu.storage("muon.grad_norm_sq", 1);
        let scratch = gpu.storage("muon.scratch", ops::scratch_len(1));

        let mut program = Program::new();
        program.extend(ops::reduce(gpu, Reduce { input: grads, in_off: 0, n_in: model.flat_len, columns: 1,
            sum: Sum::Squares, output: &grad_norm_sq, out_off: 0, accumulate: false, scratch: &scratch }));
        program.push(gpu.op_flat("muon_momentum", MUON_MOMENTUM, model.flat_len,
            Params::new().f(hyper.momentum).f(hyper.max_grad_norm),
            &[&grads.buffer, &grad_norm_sq.buffer, &momentum.buffer, &update.buffer]));

        let mut at = 0;
        while at < model.params.len() {
            let first = &model.params[at];
            let count = model.params[at..].iter().take_while(|p| p.shape == first.shape).count();
            let group = &model.params[at..at + count];
            at += count;

            // x is the update laid out wide, r by c with r the smaller
            let (r, c) = (first.rows.min(first.cols), first.rows.max(first.cols));
            let transpose = first.rows > first.cols;
            let stride = if count > 1 { group[1].off - first.off } else { 0 };
            let norm_sq = gpu.storage("muon.norm_sq", count);
            let mut x = [gpu.storage("muon.x.0", count * r * c), gpu.storage("muon.x.1", count * r * c)];
            let s = gpu.storage("muon.s", count * r * r);
            let s_squared = gpu.storage("muon.s_squared", count * r * r);
            let z = gpu.storage("muon.z", count * r * r);

            for (i, param) in group.iter().enumerate() {
                program.extend(ops::reduce(gpu, Reduce { input: &update, in_off: param.off, n_in: param.size(), columns: 1,
                    sum: Sum::Squares, output: &norm_sq, out_off: i, accumulate: false, scratch: &scratch }));
            }
            let rest = Params::new().u(first.off).u(stride).u(r * c).u(first.cols).u(c).u(transpose as usize);
            program.push(gpu.op_flat("ns_init", NS_INIT, count * r * c, rest, &[&update.buffer, &norm_sq.buffer, &x[0].buffer]));

            for (a, b, cc) in NS_COEFS {
                // s = x x^T, then x = (cc * s^2 + b * s + a * I) x
                program.extend(matmul_batched(gpu, count, r, r, c, View::rows(&x[0], 0, c).batched(r * c),
                    View::transposed(&x[0], 0, c).batched(r * c), Out::new(&s, r).batched(r * r)));
                program.extend(matmul_batched(gpu, count, r, r, r, View::rows(&s, 0, r).batched(r * r),
                    View::rows(&s, 0, r).batched(r * r), Out::new(&s_squared, r).batched(r * r)));
                program.push(gpu.op_flat("ns_poly", NS_POLY, count * r * r, Params::new().u(r).f(a).f(b).f(cc),
                    &[&s.buffer, &s_squared.buffer, &z.buffer]));
                program.extend(matmul_batched(gpu, count, r, c, r, View::rows(&z, 0, r).batched(r * r),
                    View::rows(&x[0], 0, c).batched(r * c), Out::new(&x[1], c).batched(r * c)));
                x.swap(0, 1);
            }

            let scale = (first.rows as f32 / first.cols as f32).max(1.0).sqrt();
            let rest = Params::new().u(first.off).u(stride).u(first.size()).u(first.cols).u(c).u(transpose as usize)
                .f(scale).f(hyper.weight_decay);
            program.push(gpu.op_flat("muon_apply", MUON_APPLY, count * first.size(), rest,
                &[hyper_uniform, &x[0].buffer, &model.weights.buffer]));
        }
        program
    }

    pub fn set_hyper(&self, learning_rate: f32, ent_coef: f32) {
        self.gpu.write_uniform(&self.hyper_uniform, &Params::new().f(learning_rate).f(ent_coef));
    }

    /// The program that cuts the minibatch starting at agent `lo` from the rollout.
    pub fn cut(&self, lo: usize) -> &Program {
        assert!(lo % self.segments == 0 && lo < self.agents);
        &self.cuts[lo / self.segments]
    }

    /// An epoch: `minibatches` of them, each a run of whole agents in order, as
    /// train_epoch_gpu slices them. Returns the loss statistics averaged over them.
    pub fn train(&self, learning_rate: f32, ent_coef: f32, minibatches: usize) -> [f64; 8] {
        self.set_hyper(learning_rate, ent_coef);
        self.gpu.submit(&[&self.clear_stats]);
        for mb in 0..minibatches {
            let lo = (mb * self.segments) % self.agents;
            self.gpu.submit(&[self.cut(lo), &self.forward_backward, &self.optimizer]);
        }
        self.gpu.submit(&[&self.read_stats]);
        let mut stats = [0f64; 8];
        self.gpu.read_staging(&self.stats_staging, LOSS_KEYS.len(), |sums: &[f32]| {
            for (stat, sum) in stats.iter_mut().zip(sums) {
                *stat = *sum as f64 / minibatches.max(1) as f64;
            }
        });
        stats
    }

    pub fn clear_stats(&self) {
        self.gpu.submit(&[&self.clear_stats]);
    }

    /// The loss statistics summed over the minibatches since they were cleared.
    pub fn read_stats(&self) -> Vec<f32> {
        self.gpu.read(&self.stats, 0, LOSS_KEYS.len())
    }
}
