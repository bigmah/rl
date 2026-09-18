use std::path::{Path, PathBuf};
use std::rc::Rc;

use pufferl::config::load_config;
use pufferl::gpu::Gpu;
use pufferl::pufferl::{run_eval, run_train};

const USAGE: &str = "usage: pufferl train|eval ENV [latest|MODEL.bin] [--headless] [--config FILE ...] [--section.key=value ...]";

/// Where envs/, build/ and vendor/PufferLib are: PUFFERL_ROOT, or the directory this was
/// run from, or one above the binary's own, which is where cargo leaves it.
fn root() -> PathBuf {
    let has_config = |dir: &Path| dir.join("vendor/PufferLib/config/default.ini").exists();
    if let Ok(dir) = std::env::var("PUFFERL_ROOT") {
        return PathBuf::from(dir);
    }
    let here = std::env::current_dir().unwrap_or_default();
    if has_config(&here) {
        return here;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.ancestors().find(|dir| has_config(dir)) {
            return dir.to_path_buf();
        }
    }
    pufferl::fail("vendor/PufferLib/config/default.ini not found: run this from the repository, or set PUFFERL_ROOT");
}

/// Aggressively exit on ctrl+c, as puffer does
#[cfg(unix)]
fn exit_on_interrupt() {
    extern "C" fn interrupted(_: libc::c_int) {
        unsafe { libc::_exit(0) }
    }
    unsafe {
        libc::signal(libc::SIGINT, interrupted as *const () as libc::sighandler_t);
    }
}

#[cfg(not(unix))]
fn exit_on_interrupt() {}

fn main() {
    exit_on_interrupt();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || !matches!(args[0].as_str(), "train" | "eval") {
        pufferl::fail(USAGE);
    }
    let (mode, env_name) = (args[0].as_str(), args[1].as_str());
    let mut rest: Vec<String> = args[2..].to_vec();
    let mut model = None;
    if mode == "eval" && rest.first().is_some_and(|arg| !arg.starts_with('-') && !arg.contains('=')) {
        model = Some(rest.remove(0));
    }
    let headless = rest.iter().any(|arg| arg == "--headless");
    rest.retain(|arg| arg != "--headless");

    let root = root();
    let mut ini = load_config(&root, env_name, &rest);
    if let Some(model) = model {
        ini.put("base.load_model_path", &model);
    }
    let gpu = Rc::new(Gpu::new().unwrap_or_else(|e| pufferl::fail(&e)));
    match mode {
        "train" => run_train(&root, gpu, ini),
        _ => run_eval(&root, gpu, ini, !headless),
    }
}
