//! The little of wgpu this trainer needs: buffers of 4-byte numbers, compute kernels, and
//! programs, which are lists of dispatches built once and encoded again every step.
//!
//! Every shape in the trainer is known when it starts, so every kernel's parameters and
//! bindings are too. A kernel is given its parameters in a uniform buffer of its own when its
//! op is made, and the few numbers that change while training (the step of the rollout, the
//! learning rate) live in small uniform buffers the ops share, written before a submit.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::mpsc;

/// Threads in a workgroup of a kernel that runs over a flat range.
pub const GROUP: u32 = 256;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pipelines: RefCell<HashMap<String, wgpu::ComputePipeline>>,
    allocated: Cell<u64>,
}

/// A storage buffer of `len` 4-byte numbers: f32 or u32, which is for the kernels to say.
#[derive(Clone)]
pub struct Buf {
    pub buffer: wgpu::Buffer,
    pub len: usize,
}

impl Buf {
    pub fn bytes(&self) -> u64 {
        4 * self.len as u64
    }
}

/// One dispatch of a kernel, with its parameters and buffers bound.
#[derive(Clone)]
pub struct Op {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: [u32; 3],
}

enum Step {
    Run(Op),
    Copy { src: wgpu::Buffer, src_offset: u64, dst: wgpu::Buffer, dst_offset: u64, bytes: u64 },
    Clear { buffer: wgpu::Buffer },
}

/// Dispatches and buffer copies, in the order they run.
#[derive(Default)]
pub struct Program {
    steps: Vec<Step>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, op: Op) {
        self.steps.push(Step::Run(op));
    }

    pub fn extend(&mut self, ops: impl IntoIterator<Item = Op>) {
        self.steps.extend(ops.into_iter().map(Step::Run));
    }

    /// Copy `len` numbers from `src[src_at..]` to `dst[dst_at..]`.
    pub fn copy(&mut self, src: &Buf, src_at: usize, dst: &Buf, dst_at: usize, len: usize) {
        assert!(src_at + len <= src.len && dst_at + len <= dst.len, "copy out of range");
        self.steps.push(Step::Copy {
            src: src.buffer.clone(),
            src_offset: 4 * src_at as u64,
            dst: dst.buffer.clone(),
            dst_offset: 4 * dst_at as u64,
            bytes: 4 * len as u64,
        });
    }

    pub fn clear(&mut self, buf: &Buf) {
        self.steps.push(Step::Clear { buffer: buf.buffer.clone() });
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Dispatches share a compute pass until a copy comes between them, which has to be
    /// encoded outside of one.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut i = 0;
        while i < self.steps.len() {
            match &self.steps[i] {
                Step::Run(_) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    while let Some(Step::Run(op)) = self.steps.get(i) {
                        pass.set_pipeline(&op.pipeline);
                        pass.set_bind_group(0, &op.bind_group, &[]);
                        pass.dispatch_workgroups(op.groups[0], op.groups[1], op.groups[2]);
                        i += 1;
                    }
                }
                Step::Copy { src, src_offset, dst, dst_offset, bytes } => {
                    encoder.copy_buffer_to_buffer(src, *src_offset, dst, *dst_offset, *bytes);
                    i += 1;
                }
                Step::Clear { buffer } => {
                    encoder.clear_buffer(buffer, 0, None);
                    i += 1;
                }
            }
        }
    }
}

/// A kernel's parameters: u32 and f32 fields in the order its WGSL struct declares them.
#[derive(Default, Clone)]
pub struct Params(Vec<u8>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u(mut self, value: usize) -> Self {
        let value = u32::try_from(value).expect("a kernel parameter does not fit in 32 bits");
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn f(mut self, value: f32) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }
}

impl Gpu {
    /// The first adapter wgpu offers for a high-performance device: Metal on a Mac, Vulkan or
    /// DX12 elsewhere. WGPU_BACKEND and WGPU_ADAPTER_NAME choose another, as they do for any
    /// wgpu program.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;

        // The defaults, which every WebGPU device meets, but for how big a buffer may be:
        // a rollout of pictures is bigger than the 128 MiB a binding is promised.
        let offered = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: offered.max_buffer_size,
            max_storage_buffer_binding_size: offered.max_storage_buffer_binding_size,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pufferl"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU device: {e}"))?;

        Ok(Self {
            device,
            queue,
            info: adapter.get_info(),
            pipelines: RefCell::new(HashMap::new()),
            allocated: Cell::new(0),
        })
    }

    /// Bytes of buffers made so far, which is what the dashboard calls memory used.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated.get()
    }

    fn buffer(&self, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let limit = self.device.limits().max_buffer_size;
        if bytes > limit {
            crate::fail(&format!("{label} needs {bytes} bytes and this GPU's buffers hold {limit}"));
        }
        self.allocated.set(self.allocated.get() + bytes);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Zeroed, as every wgpu buffer starts.
    pub fn storage(&self, label: &str, len: usize) -> Buf {
        let len = len.max(1);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        if 4 * len as u64 > self.device.limits().max_storage_buffer_binding_size {
            crate::fail(&format!("{label} is {len} numbers, more than this GPU binds at once"));
        }
        Buf { buffer: self.buffer(label, 4 * len as u64, usage), len }
    }

    pub fn storage_from(&self, label: &str, data: &[f32]) -> Buf {
        let buf = self.storage(label, data.len());
        self.write(&buf, 0, data);
        buf
    }

    /// For numbers that change between submits and are read by ops already built.
    pub fn uniform(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let size = (bytes.len() as u64).next_multiple_of(16).max(16);
        let buffer = self.buffer(label, size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Where the CPU reads a storage buffer's numbers from, after a copy into it.
    pub fn staging(&self, label: &str, len: usize) -> Buf {
        let len = len.max(1);
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        Buf { buffer: self.buffer(label, 4 * len as u64, usage), len }
    }

    pub fn write<T: bytemuck::Pod>(&self, buf: &Buf, at: usize, data: &[T]) {
        assert!(std::mem::size_of::<T>() == 4 && at + data.len() <= buf.len, "write out of range");
        self.queue.write_buffer(&buf.buffer, 4 * at as u64, bytemuck::cast_slice(data));
    }

    pub fn write_uniform(&self, buffer: &wgpu::Buffer, params: &Params) {
        self.queue.write_buffer(buffer, 0, &params.0);
    }

    fn pipeline(&self, name: &str, source: &str) -> wgpu::ComputePipeline {
        let key = format!("{name}\0{source}");
        if let Some(pipeline) = self.pipelines.borrow().get(&key) {
            return pipeline.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipelines.borrow_mut().insert(key, pipeline.clone());
        pipeline
    }

    /// An op: `source`'s main with `params` at binding 0 and `bindings` after it, in the
    /// order the kernel declares them, over `groups` workgroups.
    pub fn op(&self, name: &str, source: &str, params: &Params, bindings: &[&wgpu::Buffer], groups: [u32; 3]) -> Op {
        let pipeline = self.pipeline(name, source);
        let uniform = self.uniform(name, &params.0);
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
        for (i, buffer) in bindings.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: buffer.as_entire_binding() });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Op { pipeline, bind_group, groups }
    }

    /// An op over a flat range of `n` threads. Its parameters must start with `n` and `gx`,
    /// the threads across a row of the dispatch: a range longer than one row of workgroups
    /// wraps onto the next, and the kernel's index is `gid.x + gid.y * gx`.
    pub fn op_flat(&self, name: &str, source: &str, n: usize, rest: Params, bindings: &[&wgpu::Buffer]) -> Op {
        self.op_flat_layers(name, source, n, 1, rest, bindings)
    }

    /// `layers` of a flat range of `n` threads, each layer knowing which it is by `gid.z`.
    pub fn op_flat_layers(&self, name: &str, source: &str, n: usize, layers: usize, rest: Params,
            bindings: &[&wgpu::Buffer]) -> Op {
        let groups = (n as u64).div_ceil(GROUP as u64).max(1);
        let max = self.device.limits().max_compute_workgroups_per_dimension as u64;
        let across = groups.min(max);
        let down = groups.div_ceil(across);
        assert!(down <= max && layers as u64 <= max, "{name}: {n} threads in {layers} layers are more than a dispatch holds");
        let mut params = Params::new().u(n).u((across * GROUP as u64) as usize);
        params.0.extend_from_slice(&rest.0);
        self.op(name, source, &params, bindings, [across as u32, down as u32, layers as u32])
    }

    pub fn submit(&self, programs: &[&Program]) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for program in programs {
            program.encode(&mut encoder);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Wait for everything submitted so far.
    pub fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("the GPU stopped answering");
    }

    /// Map a staging buffer that a submitted copy filled, and hand its numbers to `read`.
    pub fn read_staging<T: bytemuck::Pod, R>(&self, staging: &Buf, len: usize, read: impl FnOnce(&[T]) -> R) -> R {
        let slice = staging.buffer.slice(..4 * len.max(1) as u64);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait();
        receiver.recv().expect("the GPU dropped a readback").expect("a readback could not be mapped");
        let result = {
            let view = slice.get_mapped_range().expect("a readback was not mapped");
            read(&bytemuck::cast_slice(&view)[..len])
        };
        staging.buffer.unmap();
        result
    }

    /// A storage buffer's numbers, by way of a staging buffer made for the occasion. For
    /// checkpoints and tests; the training loop keeps its staging buffers.
    pub fn read<T: bytemuck::Pod>(&self, buf: &Buf, at: usize, len: usize) -> Vec<T> {
        let staging = self.staging("read", len);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buf.buffer, 4 * at as u64, &staging.buffer, 0, 4 * len.max(1) as u64);
        self.queue.submit([encoder.finish()]);
        let out = self.read_staging(&staging, len, |data: &[T]| data.to_vec());
        self.allocated.set(self.allocated.get() - staging.bytes());
        out
    }
}
