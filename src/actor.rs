//! Acting: one step of the policy for every agent, and the rollout those steps fill.
//!
//! The rollout is laid out agent by agent, each agent's horizon of steps together, which is
//! how train_epoch_gpu transposes it: a minibatch is then a run of whole agents, and cutting
//! one is a copy.

use std::rc::Rc;

use crate::gpu::{Buf, Gpu, Params, Program};
use crate::model::Model;

const OBS_STORE: &str = include_str!("kernels/obs_store.wgsl");
const MINGRU_STEP: &str = include_str!("kernels/mingru_step.wgsl");
const SAMPLE: &str = include_str!("kernels/sample.wgsl");

/// Everything a horizon of acting leaves for training, on the GPU.
pub struct Rollout {
    pub agents: usize,
    pub horizon: usize,
    pub obs: Buf,
    pub actions: Buf,
    pub logprobs: Buf,
    pub values: Buf,
    /// Filled from the CPU once the horizon is done: the env hands these out a step at a time
    pub rewards: Buf,
    pub terminals: Buf,
    /// The MinGRU's state as the horizon began, (layers, agents, hidden)
    pub initial_state: Buf,
}

pub struct Actor {
    gpu: Rc<Gpu>,
    agents: usize,
    num_heads: usize,
    pub rollout: Rollout,
    obs: Buf,
    terminals: Buf,
    /// (layers, agents, hidden), carried from step to step and across horizons
    state: Buf,
    act_staging: Buf,
    step_uniform: wgpu::Buffer,
    seed: u32,
    counter: u32,
    keep_state: Program,
    step_and_store: Program,
    step_only: Program,
    actions: Vec<u32>,
}

impl Actor {
    pub fn new(model: &Model, agents: usize, horizon: usize, seed: u32) -> Self {
        let gpu = model.gpu.clone();
        let s = &model.shapes;
        let (hidden, layers, heads, dec_cols) = (s.hidden, s.layers, s.num_heads(), s.dec_cols());
        let slots = agents * horizon;

        let rollout = Rollout {
            agents,
            horizon,
            obs: gpu.storage("rollout.obs", slots * s.obs_size),
            actions: gpu.storage("rollout.actions", slots * heads),
            logprobs: gpu.storage("rollout.logprobs", slots),
            values: gpu.storage("rollout.values", slots),
            rewards: gpu.storage("rollout.rewards", slots),
            terminals: gpu.storage("rollout.terminals", slots),
            initial_state: gpu.storage("rollout.initial_state", layers * agents * hidden),
        };
        let obs = gpu.storage("act.obs", agents * s.obs_size);
        let terminals = gpu.storage("act.terminals", agents);
        let state = gpu.storage("act.state", layers * agents * hidden);
        let act_out = gpu.storage("act.actions", agents * heads);
        let act_staging = gpu.staging("act.actions.staging", agents * heads);
        let step_uniform = gpu.uniform("act.step", &[0u8; 16]);

        let acts = model.encoder_acts(agents, false);
        let x = [gpu.storage("act.x0", agents * hidden), gpu.storage("act.x1", agents * hidden)];
        let combined = gpu.storage("act.combined", agents * 3 * hidden);
        let dec = gpu.storage("act.dec", agents * dec_cols);

        let mut forward = model.encoder_forward(agents, &obs, &acts, &x[0]);
        for layer in 0..layers {
            let (from, to) = (&x[layer % 2], &x[(layer + 1) % 2]);
            forward.extend(model.linear(agents, from, model.layer(layer), &combined));
            let rest = Params::new().u(hidden).u(layer * agents * hidden);
            forward.push(gpu.op_flat("mingru_step", MINGRU_STEP, agents * hidden, rest,
                &[&combined.buffer, &from.buffer, &terminals.buffer, &state.buffer, &to.buffer]));
        }
        forward.extend(model.linear(agents, &x[layers % 2], model.decoder(), &dec));
        let sample = SAMPLE.replace("//HEADS", &s.heads_wgsl());
        forward.push(gpu.op_flat("sample", &sample, agents, Params::new().u(horizon),
            &[&step_uniform, &dec.buffer, &act_out.buffer, &rollout.actions.buffer,
              &rollout.logprobs.buffer, &rollout.values.buffer]));

        let store = gpu.op_flat("obs_store", OBS_STORE, agents * s.obs_size,
            Params::new().u(s.obs_size).u(horizon), &[&step_uniform, &obs.buffer, &rollout.obs.buffer]);

        let mut keep_state = Program::new();
        keep_state.copy(&state, 0, &rollout.initial_state, 0, state.len);
        let mut step_and_store = Program::new();
        step_and_store.push(store);
        let mut step_only = Program::new();
        for program in [&mut step_and_store, &mut step_only] {
            program.extend(forward.iter().cloned());
            program.copy(&act_out, 0, &act_staging, 0, agents * heads);
        }

        Self { gpu, agents, num_heads: heads, rollout, obs, terminals, state, act_staging, step_uniform,
            seed, counter: 0, keep_state, step_and_store, step_only, actions: vec![0; agents * heads] }
    }

    /// Clear the recurrent state, as a fresh episode everywhere starts from.
    pub fn clear_state(&self) {
        let mut program = Program::new();
        program.clear(&self.state);
        self.gpu.submit(&[&program]);
    }

    /// Send a step's observations and terminals, and which step of the horizon it is.
    pub fn upload(&mut self, obs: &[f32], terminals: &[f32], t: usize) {
        assert_eq!(terminals.len(), self.agents);
        self.gpu.write(&self.obs, 0, obs);
        self.gpu.write(&self.terminals, 0, terminals);
        self.gpu.write_uniform(&self.step_uniform, &Params::new().u(t).u(self.counter as usize).u(self.seed as usize));
        self.counter = self.counter.wrapping_add(1);
    }

    /// Set the policy going on what `upload` sent. With `store`, the step is kept in the
    /// rollout at `t`; at `t` 0 so is the state it began from. Nothing waits: `actions` does.
    pub fn run(&self, t: usize, store: bool) {
        if store && t == 0 {
            self.gpu.submit(&[&self.keep_state, &self.step_and_store]);
        } else if store {
            self.gpu.submit(&[&self.step_and_store]);
        } else {
            self.gpu.submit(&[&self.step_only]);
        }
    }

    /// The actions of the step `run` started, one for each head of each agent.
    pub fn actions(&mut self) -> &[u32] {
        let len = self.agents * self.num_heads;
        let actions = &mut self.actions;
        self.gpu.read_staging(&self.act_staging, len, |data: &[u32]| actions.copy_from_slice(data));
        &self.actions
    }

    /// The rewards and terminals of a finished horizon, as (agents, horizon).
    pub fn finish(&self, rewards: &[f32], terminals: &[f32]) {
        self.gpu.write(&self.rollout.rewards, 0, rewards);
        self.gpu.write(&self.rollout.terminals, 0, terminals);
    }

    /// A step's work on the GPU, for timing it apart from the way there and back.
    pub fn step_program(&self) -> &Program {
        &self.step_and_store
    }
}
