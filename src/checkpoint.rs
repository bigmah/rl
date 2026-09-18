//! Checkpoints. A 5.0 checkpoint is every weight as float32 in row order, each matrix
//! padded to a multiple of 8 numbers as the CUDA trainer's allocator lays them out, in the
//! order weights_create registers them: the encoder, the decoder, then the MinGRU. That is
//! the layout of the model's weight buffer, so PufferLib's own CPU eval (src/puffercpu.c)
//! and mlx_pufferl.py read what this writes, and this reads theirs.

use std::fs;
use std::path::{Path, PathBuf};

use crate::model::Model;

pub fn save(model: &Model, path: &Path) -> Result<(), String> {
    let weights = model.read_weights();
    let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    fs::write(&tmp, bytemuck::cast_slice::<f32, u8>(&weights)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn load(model: &Model, path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let flat = if bytes.starts_with(b"PK\x03\x04") {
        from_npz(model, &bytes).map_err(|e| format!("{}: {e}", path.display()))?
    } else {
        from_flat(model, &bytes).map_err(|e| format!("{}: {e}", path.display()))?
    };
    model.write_weights(&flat);
    Ok(())
}

fn from_flat(model: &Model, bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("{} bytes is not a whole number of float32s", bytes.len()));
    }
    let count = bytes.len() / 4;
    // The last matrix's padding is there in a checkpoint from PufferLib and from here, but
    // nothing reads it, so a file that stops where the weights do is a whole one too
    let last = model.params.last().expect("a model with no weights");
    let needed = last.off + last.size();
    if count < needed {
        return Err(format!("has {count} numbers, too few for this policy's {needed}"));
    }
    if count > model.flat_len {
        return Err(format!("has {count} numbers, more than this policy's {}", model.flat_len));
    }
    let mut flat = vec![0f32; model.flat_len];
    for (value, chunk) in flat.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(flat)
}

fn u16_at(bytes: &[u8], at: usize) -> Option<usize> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?) as usize)
}

fn u32_at(bytes: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize)
}

/// Numpy's .npz, which the PufferLib 4.0 port wrote under the same .bin names: a zip of
/// .npy arrays, stored and not compressed, each named for a weight.
fn from_npz(model: &Model, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let bad = || "not an .npz this can read".to_string();
    // The end of central directory record, then the directory it points to
    let end = (0..bytes.len().saturating_sub(21)).rev()
        .find(|&i| bytes[i..].starts_with(b"PK\x05\x06")).ok_or_else(bad)?;
    let entries = u16_at(bytes, end + 10).ok_or_else(bad)?;
    let mut at = u32_at(bytes, end + 16).ok_or_else(bad)?;

    let mut flat = vec![0f32; model.flat_len];
    let mut found = 0;
    for _ in 0..entries {
        if !bytes.get(at..).is_some_and(|rest| rest.starts_with(b"PK\x01\x02")) {
            return Err(bad());
        }
        let method = u16_at(bytes, at + 10).ok_or_else(bad)?;
        let size = u32_at(bytes, at + 24).ok_or_else(bad)?;
        let name_len = u16_at(bytes, at + 28).ok_or_else(bad)?;
        let extra_len = u16_at(bytes, at + 30).ok_or_else(bad)?;
        let comment_len = u16_at(bytes, at + 32).ok_or_else(bad)?;
        let local = u32_at(bytes, at + 42).ok_or_else(bad)?;
        let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + name_len).ok_or_else(bad)?).map_err(|_| bad())?;
        at += 46 + name_len + extra_len + comment_len;

        if method != 0 {
            return Err(format!("{name} is compressed, and this reads stored arrays only"));
        }
        let local_name = u16_at(bytes, local + 26).ok_or_else(bad)?;
        let local_extra = u16_at(bytes, local + 28).ok_or_else(bad)?;
        let start = local + 30 + local_name + local_extra;
        let npy = bytes.get(start..start + size).ok_or_else(bad)?;

        let weight = name.strip_suffix(".npy").unwrap_or(name);
        let Some(param) = model.params.iter().find(|p| p.name == weight) else {
            return Err(format!("has a weight called {weight}, which this policy has not"));
        };
        let values = npy_floats(npy).map_err(|e| format!("{name}: {e}"))?;
        if values.len() != param.size() {
            return Err(format!("{weight} has {} numbers and this policy's has {}", values.len(), param.size()));
        }
        flat[param.off..param.off + param.size()].copy_from_slice(&values);
        found += 1;
    }
    if found != model.params.len() {
        return Err(format!("has {found} weights and this policy has {}", model.params.len()));
    }
    Ok(flat)
}

fn npy_floats(npy: &[u8]) -> Result<Vec<f32>, String> {
    if !npy.starts_with(b"\x93NUMPY") || npy.len() < 12 {
        return Err("not a .npy array".into());
    }
    let (header_len, header_at) = match npy[6] {
        1 => (u16_at(npy, 8).unwrap(), 10),
        _ => (u32_at(npy, 8).unwrap(), 12),
    };
    let header = std::str::from_utf8(npy.get(header_at..header_at + header_len).ok_or("a short header")?)
        .map_err(|_| "a header that is not text")?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(format!("not little-endian float32 in row order: {}", header.trim()));
    }
    let data = &npy[header_at + header_len..];
    Ok(data.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect())
}

/// Puffer names a checkpoint after the agent step it was written at, so the most trained
/// one is the highest number and not the newest file. Picking by time loads whatever ran
/// last, which after a short run is a policy that knows nothing.
pub fn most_trained(root: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "bin") {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    found.into_iter().max_by_key(|path| {
        let steps = path.file_stem().and_then(|s| s.to_str()).filter(|s| s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<i64>().ok()).unwrap_or(-1);
        let changed = fs::metadata(path).and_then(|m| m.modified()).ok();
        (steps, changed)
    })
}
