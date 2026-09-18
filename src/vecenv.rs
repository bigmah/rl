//! The env, through vecenv.c: build/vecenv_ENV.dylib (.so, .dll), which build.sh compiles
//! around a PufferLib 5.0 C env. Observations, actions, rewards and terminals are the
//! library's own memory, read and written in place.

use std::ffi::{c_char, c_double, c_int, c_void, CString};
use std::path::Path;

use libloading::{Library, Symbol};

/// PUF_DICT_MAX_KEY
const KEY_BYTES: usize = 128;
const MAX_LOG_KEYS: usize = 64;

pub struct VecEnv {
    lib: Library,
    vec: *mut c_void,
    pub total_agents: usize,
    pub obs_size: usize,
    pub act_sizes: Vec<usize>,
    /// 1 for unsigned char observations, 4 for float
    obs_bytes: usize,
    observations: *const u8,
    actions: *mut f32,
    rewards: *const f32,
    terminals: *const f32,
}

/// Open the env's library for good. It steps envs on OpenMP threads, which outlive every
/// call into it and wait between calls inside the OpenMP runtime the library brought in.
/// Unloading the library as the trainer exits unloads that runtime under them, and the
/// first to wake returns into memory that is gone: with GCC's libgomp on Linux, a
/// segfault at the end of about one run in ten. RTLD_NODELETE keeps it mapped until the
/// process ends, which is when its threads do. The rest are libloading's own flags.
#[cfg(unix)]
unsafe fn open(path: &Path) -> Result<Library, libloading::Error> {
    let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_NODELETE;
    libloading::os::unix::Library::open(Some(path), flags).map(Library::from)
}

#[cfg(not(unix))]
unsafe fn open(path: &Path) -> Result<Library, libloading::Error> {
    Library::new(path)
}

pub fn library_name(env_name: &str) -> String {
    let extension = if cfg!(target_os = "macos") { "dylib" } else if cfg!(windows) { "dll" } else { "so" };
    format!("vecenv_{env_name}.{extension}")
}

impl VecEnv {
    pub fn new(root: &Path, env_name: &str, env_config: &[(String, String)], total_agents: usize, num_threads: usize) -> Self {
        let path = root.join("build").join(library_name(env_name));
        if !path.exists() {
            crate::fail(&format!("{} not found: run ./build.sh {env_name}", path.display()));
        }
        // The library is an env someone compiled to be trained on; loading it runs it
        let lib = unsafe { open(&path) }.unwrap_or_else(|e| crate::fail(&format!("{}: {e}", path.display())));

        let keys: Vec<CString> = env_config.iter().map(|(k, _)| CString::new(k.as_str()).unwrap()).collect();
        let values: Vec<CString> = env_config.iter().map(|(_, v)| CString::new(v.as_str()).unwrap()).collect();
        let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
        let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();

        unsafe {
            let create: Symbol<unsafe extern "C" fn(c_int, c_int, c_int, *const *const c_char, *const *const c_char) -> *mut c_void>
                = lib.get(b"vec_create").unwrap_or_else(|e| crate::fail(&format!("{}: {e}", path.display())));
            let vec = create(total_agents as c_int, num_threads as c_int, keys.len() as c_int, key_ptrs.as_ptr(), value_ptrs.as_ptr());

            let int = |name: &[u8]| -> usize {
                let f: Symbol<unsafe extern "C" fn() -> c_int> = lib.get(name).expect("vecenv symbol");
                f() as usize
            };
            // Only final once the envs are made: an env may size its observation from its config
            let obs_size = int(b"vec_obs_size");
            let num_atns = int(b"vec_num_atns");
            let obs_bytes = int(b"vec_obs_bytes");
            let sizes: Symbol<unsafe extern "C" fn() -> *const c_int> = lib.get(b"vec_act_sizes").expect("vecenv symbol");
            let act_sizes = std::slice::from_raw_parts(sizes(), num_atns).iter().map(|&a| a as usize).collect();

            let ptr = |name: &[u8]| -> *mut c_void {
                let f: Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> = lib.get(name).expect("vecenv symbol");
                f(vec)
            };
            let observations = ptr(b"vec_observations") as *const u8;
            let actions = ptr(b"vec_actions") as *mut f32;
            let rewards = ptr(b"vec_rewards") as *const f32;
            let terminals = ptr(b"vec_terminals") as *const f32;
            Self { lib, vec, total_agents, obs_size, act_sizes, obs_bytes, observations, actions, rewards, terminals }
        }
    }

    fn call(&self, name: &[u8]) {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut c_void)> = self.lib.get(name).expect("vecenv symbol");
            f(self.vec);
        }
    }

    pub fn reset(&mut self) {
        self.call(b"vec_reset");
    }

    /// One step of every env, on the actions in place.
    pub fn step(&mut self) {
        self.call(b"vec_step");
    }

    /// Every agent's observation as floats, which is what the policy's first matrix takes.
    pub fn observations(&self, out: &mut Vec<f32>) {
        let count = self.total_agents * self.obs_size;
        out.clear();
        unsafe {
            if self.obs_bytes == 1 {
                out.extend(std::slice::from_raw_parts(self.observations, count).iter().map(|&b| b as f32));
            } else {
                out.extend_from_slice(std::slice::from_raw_parts(self.observations as *const f32, count));
            }
        }
    }

    pub fn rewards(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.rewards, self.total_agents) }
    }

    pub fn terminals(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.terminals, self.total_agents) }
    }

    /// The env reads its actions as floats, one for each head of each agent.
    pub fn set_actions(&mut self, actions: &[u32]) {
        let out = unsafe { std::slice::from_raw_parts_mut(self.actions, self.total_agents * self.act_sizes.len()) };
        for (slot, &action) in out.iter_mut().zip(actions) {
            *slot = action as f32;
        }
    }

    /// Env stats averaged over the episodes since the last clear, and n, how many
    pub fn log(&mut self, clear: bool) -> Vec<(String, f64)> {
        let mut keys = vec![0u8; MAX_LOG_KEYS * KEY_BYTES];
        let mut values = [0f64; MAX_LOG_KEYS];
        let count = unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut c_char, *mut c_double) -> c_int>
                = self.lib.get(b"vec_log").expect("vecenv symbol");
            f(self.vec, clear as c_int, MAX_LOG_KEYS as c_int, keys.as_mut_ptr() as *mut c_char, values.as_mut_ptr())
        } as usize;
        (0..count).map(|i| {
            let key = &keys[i * KEY_BYTES..(i + 1) * KEY_BYTES];
            let end = key.iter().position(|&b| b == 0).unwrap_or(KEY_BYTES);
            (String::from_utf8_lossy(&key[..end]).into_owned(), values[i])
        }).collect()
    }

    pub fn render(&mut self, env_id: usize) {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = self.lib.get(b"vec_render").expect("vecenv symbol");
            f(self.vec, env_id as c_int);
        }
    }

    pub fn window_closed(&self) -> bool {
        unsafe {
            let f: Symbol<unsafe extern "C" fn() -> c_int> = self.lib.get(b"vec_window_closed").expect("vecenv symbol");
            f() != 0
        }
    }

    pub fn close(&mut self) {
        if !self.vec.is_null() {
            self.call(b"vec_close");
            self.vec = std::ptr::null_mut();
        }
    }
}

impl Drop for VecEnv {
    fn drop(&mut self) {
        self.close();
    }
}
