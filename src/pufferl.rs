//! The trainer (src/pufferl.cu): rollouts, epochs of training on them, logs, and the train
//! and eval runs around those (run_train, run_eval).

use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::actor::Actor;
use crate::checkpoint;
use crate::config::Ini;
use crate::dashboard::Dashboard;
use crate::gpu::Gpu;
use crate::learner::{Hyper, Learner, LOSS_KEYS};
use crate::model::{Model, Shapes};
use crate::vecenv::VecEnv;

const PERF_KEYS: [&str; 6] = ["rollout", "eval_model", "eval_env", "eval_copy", "train_misc", "train_model"];

/// What a log says, in the order it was said: the dashboard shows an env's stats in the
/// order the env gave them.
#[derive(Clone, Debug, Default)]
pub struct Log(pub Vec<(String, f64)>);

impl Log {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    pub fn or(&self, key: &str, default: f64) -> f64 {
        self.get(key).unwrap_or(default)
    }

    pub fn set(&mut self, key: &str, value: f64) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value,
            None => self.0.push((key.to_string(), value)),
        }
    }
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn cosine_annealing(base: f64, minimum: f64, epoch: usize, total_epochs: usize) -> f64 {
    minimum + 0.5 * (base - minimum) * (1.0 + (std::f64::consts::PI * epoch as f64 / total_epochs as f64).cos())
}

/// The most memory this process has held, in bytes: getrusage counts it in bytes on a Mac
/// and in kilobytes everywhere else.
fn peak_memory_bytes() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let scale = if cfg!(target_os = "macos") { 1.0 } else { 1024.0 };
        usage.ru_maxrss as f64 * scale
    }
    #[cfg(not(unix))]
    0.0
}

/// What the GPU has to work with, where that is something this can know: a GPU that shares
/// the machine's memory has all of it, and a card's own memory wgpu does not tell.
fn gpu_memory_bytes(gpu: &Gpu) -> f64 {
    #[cfg(unix)]
    if gpu.info.device_type == wgpu::DeviceType::IntegratedGpu {
        unsafe {
            let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
            let page = libc::sysconf(libc::_SC_PAGESIZE);
            if pages > 0 && page > 0 {
                return pages as f64 * page as f64;
            }
        }
    }
    let _ = gpu;
    0.0
}

pub struct PuffeRL {
    pub ini: Ini,
    pub gpu: Rc<Gpu>,
    pub vec: VecEnv,
    pub model: Model,
    actor: Actor,
    learner: Option<Learner>,
    pub total_agents: usize,
    pub horizon: usize,
    pub batch_size: usize,
    pub total_epochs: usize,
    reset_every_horizon: bool,
    /// (agents, horizon), as the rollout is
    rewards: Vec<f32>,
    terminals: Vec<f32>,
    obs: Vec<f32>,
    /// With target_entropy set, ent_coef is only where the coefficient starts: see train()
    ent_coef: f64,
    log_ent_coef: f64,
    current_ent_coef: f64,
    pub epoch: usize,
    pub global_step: usize,
    start_time: f64,
    pub last_log_time: f64,
    last_log_step: usize,
    loss_sums: [f64; 8],
    loss_count: usize,
    perf: [f64; 6],
}

impl PuffeRL {
    /// The env, the policy for it, and a checkpoint into the policy if the config names one.
    /// Eval has no use for a learner, which is most of the memory.
    pub fn create(root: &Path, gpu: Rc<Gpu>, ini: &Ini, training: bool) -> Self {
        let env_name = ini.text("base", "env_name");
        let total_agents = ini.size("vec", "total_agents");
        let vec = VecEnv::new(root, &env_name, ini.section("env"), total_agents, ini.size("vec", "num_threads"));
        if vec.act_sizes.iter().sum::<usize>() == vec.act_sizes.len() {
            crate::fail("pufferl only implements discrete actions");
        }

        // An env that shows its policy a picture puts it last in the observation, RGB, and
        // says how big it is in its own config
        let width = ini.num_opt("env", "picture_width", 0.0) as usize;
        let height = ini.num_opt("env", "picture_height", 0.0) as usize;
        let picture = (width > 0 && height > 0).then_some((height, width, 3));
        if picture.is_some() && vec.obs_size < height * width * 3 {
            crate::fail(&format!("an observation of {} has no room for a {width}x{height} picture", vec.obs_size));
        }
        let shapes = Shapes {
            obs_size: vec.obs_size,
            act_sizes: vec.act_sizes.clone(),
            hidden: ini.size("policy", "hidden_size"),
            layers: ini.size("policy", "num_layers"),
            picture,
        };
        let seed = ini.int("base", "seed") as u64;
        let model = Model::new(gpu.clone(), shapes, seed);

        let horizon = ini.size("train", "horizon");
        let actor = Actor::new(&model, total_agents, horizon, seed as u32);
        let learner = training.then(|| {
            let segments = ini.size("train", "minibatch_size") / horizon;
            let f = |key: &str| ini.num("train", key) as f32;
            Learner::new(&model, &actor.rollout, segments, &Hyper {
                gamma: f("gamma"),
                gae_lambda: f("gae_lambda"),
                clip_coef: f("clip_coef"),
                vf_coef: f("vf_coef"),
                vf_clip_coef: f("vf_clip_coef"),
                max_grad_norm: f("max_grad_norm"),
                momentum: f("momentum"),
                weight_decay: ini.num_opt("train", "weight_decay", 0.0) as f32,
                vtrace: ini.flag("train", "vtrace"),
                vtrace_rho_clip: f("vtrace_rho_clip"),
                vtrace_c_clip: f("vtrace_c_clip"),
                norm_adv: ini.flag("train", "norm_adv"),
            })
        });

        if let Some(path) = checkpoint_path(ini) {
            println!("Loading {}", path.display());
            checkpoint::load(&model, &path).unwrap_or_else(|e| crate::fail(&e));
        }

        let batch_size = total_agents * horizon;
        let ent_coef = ini.num("train", "ent_coef");
        let mut pufferl = Self {
            ini: ini.clone(),
            gpu,
            vec,
            model,
            actor,
            learner,
            total_agents,
            horizon,
            batch_size,
            total_epochs: (ini.size("train", "total_timesteps") / batch_size).max(1),
            reset_every_horizon: ini.flag("base", "reset_every_horizon"),
            rewards: vec![0.0; batch_size],
            terminals: vec![0.0; batch_size],
            obs: Vec::new(),
            ent_coef,
            log_ent_coef: if ent_coef > 0.0 { ent_coef.ln() } else { 0.0 },
            current_ent_coef: ent_coef,
            epoch: 0,
            global_step: 0,
            start_time: now(),
            last_log_time: now(),
            last_log_step: 0,
            loss_sums: [0.0; 8],
            loss_count: 0,
            perf: [0.0; 6],
        };
        pufferl.vec.reset();
        pufferl
    }

    pub fn num_params(&self) -> usize {
        self.model.num_params
    }

    /// horizon steps of every agent. At each: the observation, and the reward and terminal
    /// the last step left, are stored with the action taken from them
    pub fn rollouts(&mut self, store: bool) {
        let start = Instant::now();
        let (agents, horizon) = (self.total_agents, self.horizon);
        if self.reset_every_horizon {
            self.actor.clear_state();
        }

        for t in 0..horizon {
            let t0 = Instant::now();
            self.vec.observations(&mut self.obs);
            if store {
                for (agent, (&reward, &terminal)) in self.vec.rewards().iter().zip(self.vec.terminals()).enumerate() {
                    self.rewards[agent * horizon + t] = reward;
                    self.terminals[agent * horizon + t] = terminal;
                }
            }
            self.actor.upload(&self.obs, self.vec.terminals(), t);
            let t1 = Instant::now();
            self.actor.run(t, store);
            let actions = self.actor.actions();
            let t2 = Instant::now();
            self.vec.set_actions(actions);
            let t3 = Instant::now();
            self.vec.step();
            self.perf[3] += (t1 - t0).as_secs_f64() + (t3 - t2).as_secs_f64();
            self.perf[1] += (t2 - t1).as_secs_f64();
            self.perf[2] += t3.elapsed().as_secs_f64();
        }

        self.global_step += agents * horizon;
        self.perf[0] += start.elapsed().as_secs_f64();
    }

    pub fn train(&mut self) {
        let start = Instant::now();
        let ini = &self.ini;
        let learning_rate = ini.num("train", "learning_rate");
        let lr = if ini.flag("train", "anneal_lr") {
            cosine_annealing(learning_rate, ini.num("train", "min_lr_ratio") * learning_rate, self.epoch, self.total_epochs)
        } else {
            learning_rate
        };
        let mut ent_coef = ini.num("train", "ent_coef");
        let target_entropy = ini.num_opt("train", "target_entropy", 0.0);
        if target_entropy != 0.0 {
            ent_coef = self.ent_coef;
        } else if ini.flag("train", "anneal_ent_coef") {
            ent_coef = cosine_annealing(ent_coef, ini.num("train", "min_ent_coef_ratio") * ent_coef, self.epoch, self.total_epochs);
        }
        self.current_ent_coef = ent_coef;

        for reward in &mut self.rewards {
            *reward = reward.clamp(-1.0, 1.0);
        }
        self.actor.finish(&self.rewards, &self.terminals);
        let model_start = Instant::now();

        // Minibatches are runs of whole rows in agent order, as train_epoch_gpu slices them
        let minibatches = (ini.num("train", "replay_ratio") * self.batch_size as f64 / ini.num("train", "minibatch_size")) as usize;
        let learner = self.learner.as_ref().expect("this PuffeRL was made for eval");
        let stats = learner.train(lr as f32, ent_coef as f32, minibatches);
        for (sum, stat) in self.loss_sums.iter_mut().zip(stats) {
            *sum += stat * minibatches as f64;
        }
        self.loss_count += minibatches;

        if target_entropy != 0.0 && minibatches > 0 {
            // Steer the coefficient toward the entropy asked for: up while the policy is
            // more certain than that, down while it is less, by a factor of e every
            // 1 / (ent_coef_rate * gap) epochs, kept between 1e-4 and 0.5.
            let (lo, hi) = (1e-4f64.ln(), 0.5f64.ln());
            let gap = target_entropy - stats[2];
            self.log_ent_coef = (self.log_ent_coef + ini.num_opt("train", "ent_coef_rate", 0.01) * gap).clamp(lo, hi);
            // That sum alone overshoots where entropy answers the coefficient late: it keeps
            // climbing until entropy has crossed the target, and by then is far past what holds
            // it there. ent_coef_damping adds the gap itself, a factor of e for every
            // 1 / ent_coef_damping of entropy off target, which is gone the moment the gap is.
            let damped = self.log_ent_coef + ini.num_opt("train", "ent_coef_damping", 0.0) * gap;
            self.ent_coef = damped.clamp(lo, hi).exp();
        }
        self.epoch += 1;

        self.perf[4] += (model_start - start).as_secs_f64();
        self.perf[5] += model_start.elapsed().as_secs_f64();
    }

    /// The keys run_train logs, under the names pufferl.cu gives them
    pub fn log(&mut self, clear: bool) -> Log {
        let time = now();
        let dt = time - self.last_log_time;
        let mut log = Log::default();
        log.set("SPS", if dt > 0.0 { (self.global_step - self.last_log_step) as f64 / dt } else { 0.0 });
        log.set("agent_steps", self.global_step as f64);
        log.set("uptime", time - self.start_time);
        log.set("epoch", self.epoch as f64);
        self.last_log_time = time;
        self.last_log_step = self.global_step;
        for (key, value) in self.vec.log(clear) {
            log.set(&format!("env/{key}"), value);
        }
        if self.loss_count > 0 {
            for (key, sum) in LOSS_KEYS.iter().zip(self.loss_sums) {
                log.set(key, sum / self.loss_count as f64);
            }
            log.set("train/ent_coef", self.current_ent_coef);
        }
        self.loss_sums = [0.0; 8];
        self.loss_count = 0;
        log.set("util/vram_used_gb", self.gpu.allocated_bytes() as f64 / 1e9);
        log.set("util/vram_total_gb", gpu_memory_bytes(&self.gpu) / 1e9);
        log.set("util/cpu_mem_gb", peak_memory_bytes() / 1e9);
        for (key, seconds) in PERF_KEYS.iter().zip(self.perf) {
            log.set(&format!("perf/{key}"), seconds);
        }
        log.set("perf/train", self.perf[4] + self.perf[5]);
        self.perf = [0.0; 6];
        log
    }

    /// Fresh episodes and a clear recurrent state, as env_restart gives eval
    pub fn restart(&mut self) {
        self.vec.reset();
        self.actor.clear_state();
        self.vec.log(true);
    }

    pub fn close(&mut self) {
        self.vec.close();
    }
}

fn checkpoint_path(ini: &Ini) -> Option<PathBuf> {
    let load_path = ini.text("base", "load_model_path");
    match load_path.as_str() {
        "None" | "" => None,
        "latest" => {
            let root = Path::new(&ini.text("base", "checkpoint_dir")).join(ini.text("base", "env_name"));
            Some(checkpoint::most_trained(&root)
                .unwrap_or_else(|| crate::fail(&format!("no .bin checkpoints found in {}/", root.display()))))
        }
        path => Some(PathBuf::from(path)),
    }
}

/// log_history_bin_mean: history[key] averaged over points bins of agent_steps, the last
/// bin being the final value
fn bin_mean(history: &[&Log], key: &str, points: usize) -> Vec<f64> {
    let value = |log: &Log| log.or(key, 0.0);
    let last = value(history[history.len() - 1]);
    if points == 1 {
        return vec![last];
    }
    let final_steps = history[history.len() - 1].or("agent_steps", 0.0);
    let step = final_steps / (points - 1) as f64;
    let (mut out, mut bin_sum, mut bin_n) = (Vec::new(), 0.0, 0usize);
    let mut fallback = value(history[0]);
    let mut next_bin = step;
    for log in history {
        bin_sum += value(log);
        bin_n += 1;
        if log.or("agent_steps", 0.0) < next_bin || out.len() >= points - 1 {
            continue;
        }
        fallback = bin_sum / bin_n as f64;
        out.push(fallback);
        (bin_sum, bin_n) = (0.0, 0);
        next_bin += step;
    }
    while out.len() < points - 1 {
        out.push(fallback);
    }
    out.push(last);
    out
}

/// C's %.17g, which is how the log writes a number so that it reads back the same.
fn g17(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    if !value.is_finite() {
        return if value.is_nan() { "nan".into() } else if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let scientific = format!("{value:.16e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let strip = |s: &str| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
    };
    if !(-4..17).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{}e{sign}{:02}", strip(mantissa), exponent.abs());
    }
    strip(&format!("{value:.*}", (16 - exponent).max(0) as usize))
}

/// Rollouts with no training until episodes have ended, or until the window closes when
/// rendering. With a board, env stats are merged into it for the dashboard
fn eval_loop(pufferl: &mut PuffeRL, dashboard: &mut Dashboard, render: bool, episodes: f64, mut board: Option<&mut Log>) -> Log {
    let mut last_dash = 0.0;
    loop {
        if render {
            pufferl.vec.render(0);
            if pufferl.vec.window_closed() {
                return Log::default();
            }
        }
        pufferl.rollouts(false);
        if render {
            continue;
        }
        let log = pufferl.log(false);
        if let Some(board) = board.as_deref_mut() {
            for (key, value) in log.0.iter().filter(|(key, _)| key.starts_with("env/")) {
                board.set(key, *value);
            }
        }
        let n = log.or("env/n", 0.0);
        if n >= episodes || now() - last_dash >= 0.6 {
            dashboard.print(pufferl, board.as_deref().unwrap_or(&log), pufferl.epoch);
            last_dash = now();
        }
        if n >= episodes {
            return log;
        }
    }
}

pub fn run_train(root: &Path, gpu: Rc<Gpu>, mut ini: Ini) {
    let (minibatch, horizon, agents) = (ini.size("train", "minibatch_size"), ini.size("train", "horizon"), ini.size("vec", "total_agents"));
    if horizon == 0 || minibatch % horizon != 0 {
        crate::fail("train.minibatch_size must be divisible by train.horizon");
    }
    if minibatch > horizon * agents {
        crate::fail("train.minibatch_size must be <= train.horizon * vec.total_agents");
    }
    if agents % (minibatch / horizon) != 0 {
        crate::fail("vec.total_agents must be divisible by minibatch rows");
    }
    if matches!(ini.text("base", "run_id").as_str(), "None" | "") {
        ini.put("base.run_id", &((1000.0 * now()) as u64).to_string());
    }
    let (env_name, run_id) = (ini.text("base", "env_name"), ini.text("base", "run_id"));
    let checkpoint_dir = Path::new(&ini.text("base", "checkpoint_dir")).join(&env_name).join(&run_id);
    let log_dir = Path::new(&ini.text("base", "log_dir")).join(&env_name);
    for dir in [&checkpoint_dir, &log_dir] {
        fs::create_dir_all(dir).unwrap_or_else(|e| crate::fail(&format!("{}: {e}", dir.display())));
    }

    let mut pufferl = PuffeRL::create(root, gpu.clone(), &ini, true);
    let mut dashboard = Dashboard::new();
    let train_epochs = ini.size("train", "total_timesteps") / pufferl.batch_size;
    let interval = ini.int("base", "checkpoint_interval");
    let target_key = format!("env/{}", ini.text("sweep", "metric"));
    let mut last_log: Option<Log> = None;
    let mut history: Vec<Log> = Vec::new();
    for epoch in 0..train_epochs {
        pufferl.rollouts(true);
        pufferl.train();
        if epoch == train_epochs - 1 || (interval > 0 && (epoch + 1) % interval as usize == 0) {
            let path = checkpoint_dir.join(format!("{:016}.bin", pufferl.global_step));
            checkpoint::save(&pufferl.model, &path).unwrap_or_else(|e| crate::fail(&e));
        }
        if last_log.is_some() && now() < pufferl.last_log_time + 0.6 && epoch < train_epochs - 1 {
            continue;
        }
        let log = pufferl.log(true);
        dashboard.print(&pufferl, &log, pufferl.epoch);
        // A log with no episodes in it has no env stats, so keep the last one that did
        let episodes = log.or("env/n", 0.0) > 0.0;
        match &mut last_log {
            Some(last) if !episodes => {
                last.set("uptime", log.or("uptime", 0.0));
                last.set("agent_steps", log.or("agent_steps", 0.0));
            }
            _ => last_log = Some(log.clone()),
        }
        if episodes && log.get(&target_key).is_some() {
            history.push(last_log.clone().unwrap());
        }
    }
    let mut last_log = last_log.unwrap_or_default();

    let eval_episodes = ini.num("base", "eval_episodes");
    if eval_episodes > 0.0 && train_epochs > 0 {
        let eval_agents = ini.int("base", "eval_agents");
        if eval_agents != -1 {
            let weights = pufferl.model.read_weights();
            pufferl.close();
            drop(pufferl);
            ini.put("vec.total_agents", &eval_agents.to_string());
            pufferl = PuffeRL::create(root, gpu, &ini, false);
            pufferl.model.write_weights(&weights);
        }
        pufferl.restart();
        eval_loop(&mut pufferl, &mut dashboard, false, eval_episodes, Some(&mut last_log));
    }
    pufferl.close();

    history.push(last_log.clone());
    let points = ini.size("sweep", "downsample");
    let mut out = String::from("# PufferLib log v1\n");
    ini.write(&mut out);
    out.push_str("\n[metrics]\n");
    if last_log.get(&target_key).is_some() {
        for (key, _) in last_log.0.iter().filter(|(key, _)| !key.starts_with("loss/")) {
            let with_key: Vec<&Log> = history.iter().filter(|log| log.get(key).is_some()).collect();
            let values: Vec<String> = bin_mean(&with_key, key, points).into_iter().map(g17).collect();
            out.push_str(&format!("{key} = {}\n", values.join(",")));
        }
    }
    let path = log_dir.join(format!("{run_id}.ini"));
    fs::write(&path, out).unwrap_or_else(|e| crate::fail(&format!("{}: {e}", path.display())));
}

pub fn run_eval(root: &Path, gpu: Rc<Gpu>, mut ini: Ini, render: bool) {
    if render {
        // One env on screen, a step at a time
        let agents = ini.text("env", "num_agents");
        ini.put("vec.total_agents", &agents);
        ini.put("vec.num_threads", "1");
        ini.put("train.horizon", "1");
    }
    ini.put("base.reset_every_horizon", "0");
    let episodes = ini.num("base", "eval_episodes");
    let mut pufferl = PuffeRL::create(root, gpu, &ini, false);
    let log = eval_loop(&mut pufferl, &mut Dashboard::new(), render, episodes, None);
    if !render {
        println!("EVAL env={} score={:.6} perf={:.6} games={} params={}", ini.text("base", "env_name"),
            log.or("env/score", 0.0), log.or("env/perf", 0.0), log.or("env/n", 0.0) as i64, pufferl.num_params());
    }
    pufferl.close();
}

#[cfg(test)]
mod tests {
    use super::g17;

    #[test]
    fn seventeen_digits() {
        assert_eq!(g17(0.5), "0.5");
        assert_eq!(g17(65536.0), "65536");
        assert_eq!(g17(0.1), "0.10000000000000001");
        assert_eq!(g17(1e-5), "1.0000000000000001e-05");
        assert_eq!(g17(1e17), "1e+17");
        assert_eq!(g17(-2.25), "-2.25");
    }
}
