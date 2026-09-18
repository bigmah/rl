//! PufferLib 5.0's trainer on wgpu, for whatever GPU a machine has.
//!
//! PufferLib 5.0 trains with one CUDA program (src/pufferl.cu, src/algo.cu). This is that
//! trainer on wgpu, which is Metal on a Mac, Vulkan on Linux and DX12 on Windows: the same
//! policy (bias-free Linear encoder -> MinGRU -> one decoder matrix for logits and value),
//! the same PPO (advantages from each minibatch's own values, no prioritized replay,
//! recurrent state carried across horizons), Muon, the same config files and CLI overrides,
//! checkpoints PufferLib's own eval can read, and its dashboard. Like the CUDA trainer it
//! has no autograd: every backward pass is a kernel written for it (src/kernels).
//!
//! Envs are PufferLib 5.0 C envs, which build.sh compiles into a vecenv library (vecenv.c)
//! that this loads:
//!
//! ```text
//! pufferl train platformer [--section.key=value ...]
//! pufferl eval platformer [latest | MODEL.bin] [--headless] [--section.key=value ...]
//! ```

pub mod actor;
pub mod checkpoint;
pub mod config;
pub mod dashboard;
pub mod gpu;
pub mod learner;
pub mod model;
pub mod ops;
pub mod pufferl;
pub mod rng;
pub mod vecenv;

/// Say what is wrong and stop, as sys.exit does with a message.
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
