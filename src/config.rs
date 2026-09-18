//! Config: src/ini.h's reading of default.ini, the env's ini, and the command line.

use std::fs;
use std::path::Path;

/// Sections in the order they were first seen, each with its keys in theirs, and every
/// value as it was written: what a run's log records.
#[derive(Clone, Debug, Default)]
pub struct Ini {
    pub sections: Vec<(String, Vec<(String, String)>)>,
}

/// A value the way ini.h reads one.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    List(Vec<f64>),
    Str(String),
}

fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut prev = '\0';
    for (i, ch) in line.char_indices() {
        if (ch == '\'' || ch == '"') && prev != '\\' {
            quote = match quote {
                Some(open) if open == ch => None,
                Some(open) => Some(open),
                None => Some(ch),
            };
        } else if (ch == '#' || ch == ';') && quote.is_none() {
            return &line[..i];
        }
        prev = ch;
    }
    line
}

fn parse_float(raw: &str) -> Option<f64> {
    let cleaned: String = raw.chars().filter(|ch| *ch != '_' && !ch.is_whitespace()).collect();
    // Takes "infinity" and "nan", but not a bare sign or an empty string
    cleaned.parse::<f64>().ok()
}

/// true and false, numbers with underscores, and lists
pub fn parse(raw: &str) -> Value {
    match raw.to_ascii_lowercase().as_str() {
        "true" => return Value::Int(1),
        "false" => return Value::Int(0),
        _ => {}
    }
    if raw.contains(',') {
        let parts: Option<Vec<f64>> = raw.split(',').map(|part| part.trim().parse::<f64>().ok()).collect();
        return parts.map(Value::List).unwrap_or_else(|| Value::Str(raw.to_string()));
    }
    match parse_float(raw) {
        Some(value) if value.fract() == 0.0 && value.abs() < 2f64.powi(62) => Value::Int(value as i64),
        Some(value) => Value::Float(value),
        None => Value::Str(raw.to_string()),
    }
}

impl Ini {
    fn section_mut(&mut self, name: &str) -> &mut Vec<(String, String)> {
        if let Some(i) = self.sections.iter().position(|(section, _)| section == name) {
            return &mut self.sections[i].1;
        }
        self.sections.push((name.to_string(), Vec::new()));
        &mut self.sections.last_mut().unwrap().1
    }

    pub fn section(&self, name: &str) -> &[(String, String)] {
        self.sections.iter().find(|(section, _)| section == name).map(|(_, keys)| keys.as_slice()).unwrap_or(&[])
    }

    pub fn load(&mut self, path: &Path) {
        let text = fs::read_to_string(path).unwrap_or_else(|e| crate::fail(&format!("{}: {e}", path.display())));
        let mut current: Option<String> = None;
        for (n, line) in text.lines().enumerate() {
            let s = strip_comment(line).trim();
            if s.is_empty() {
                continue;
            }
            if s.starts_with('[') && s.ends_with(']') && s.len() >= 3 {
                let name = s[1..s.len() - 1].trim().to_string();
                self.section_mut(&name);
                current = Some(name);
                continue;
            }
            let Some(section) = &current else {
                crate::fail(&format!("{}:{}: expected section before key=value", path.display(), n + 1));
            };
            let Some((key, value)) = s.split_once('=') else {
                crate::fail(&format!("{}:{}: expected key=value", path.display(), n + 1));
            };
            let (key, mut value) = (key.trim(), value.trim());
            let quoted = value.len() >= 2 && (value.starts_with('\'') || value.starts_with('"'))
                && value.chars().next() == value.chars().last();
            if quoted {
                value = &value[1..value.len() - 1];
            }
            let keys = self.section_mut(section);
            match keys.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value.to_string(),
                None => keys.push((key.to_string(), value.to_string())),
            }
        }
    }

    pub fn raw(&self, section: &str, key: &str) -> Option<&str> {
        self.section(section).iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Only keys a config file already has can be set, as in ini.h
    pub fn put(&mut self, full_key: &str, raw: &str) {
        let (section, key) = full_key.rsplit_once('.').unwrap_or(("", full_key));
        let entry = self.sections.iter_mut().find(|(name, _)| name == section)
            .and_then(|(_, keys)| keys.iter_mut().find(|(k, _)| k == key));
        match entry {
            Some(entry) => entry.1 = raw.to_string(),
            None => crate::fail(&format!("missing key [{section}] {key}")),
        }
    }

    pub fn get(&self, section: &str, key: &str) -> Option<Value> {
        self.raw(section, key).map(parse)
    }

    /// A number a config has to have.
    pub fn num(&self, section: &str, key: &str) -> f64 {
        self.num_or(section, key, f64::NAN, true)
    }

    /// A number a config may leave out, as [train] may the keys PufferLib has not got.
    pub fn num_opt(&self, section: &str, key: &str, default: f64) -> f64 {
        self.num_or(section, key, default, false)
    }

    fn num_or(&self, section: &str, key: &str, default: f64, required: bool) -> f64 {
        match self.get(section, key) {
            Some(Value::Int(v)) => v as f64,
            Some(Value::Float(v)) => v,
            Some(other) => crate::fail(&format!("[{section}] {key} should be a number, not {other:?}")),
            None if required => crate::fail(&format!("missing key [{section}] {key}")),
            None => default,
        }
    }

    pub fn int(&self, section: &str, key: &str) -> i64 {
        self.num(section, key) as i64
    }

    pub fn size(&self, section: &str, key: &str) -> usize {
        let value = self.num(section, key);
        if value < 0.0 {
            crate::fail(&format!("[{section}] {key} cannot be negative"));
        }
        value as usize
    }

    pub fn flag(&self, section: &str, key: &str) -> bool {
        self.num_opt(section, key, 0.0) != 0.0
    }

    pub fn text(&self, section: &str, key: &str) -> String {
        match self.raw(section, key) {
            Some(raw) => raw.to_string(),
            None => crate::fail(&format!("missing key [{section}] {key}")),
        }
    }

    /// Another config file's values over these. Like a flag, it can only set keys this one
    /// already has, so a misspelt key is an error rather than a setting nobody reads.
    pub fn overlay(&mut self, path: &Path) {
        let mut other = Ini::default();
        other.load(path);
        for (section, keys) in other.sections {
            for (key, raw) in keys {
                if self.raw(&section, &key).is_none() {
                    crate::fail(&format!("{}: [{section}] {key} is not a key the config has", path.display()));
                }
                self.put(&format!("{section}.{key}"), &raw);
            }
        }
    }

    pub fn write(&self, out: &mut String) {
        for (section, keys) in &self.sections {
            out.push_str(&format!("\n[{section}]\n"));
            for (key, raw) in keys {
                out.push_str(&format!("{key} = {raw}\n"));
            }
        }
    }
}

/// default.ini, then envs/ENV/ENV.ini, then each --config FILE in the order given, then
/// --section.key=value (or --section.key value) from the command line, wherever they are in
/// it. A key with no section is in [base], and dashes in a key are underscores.
///
/// A --config file is a variant of the env's config -- sm64's stars/pss-2.ini, the one star's
/// clock and respawn -- and may only set keys the configs before it have, as a flag may.
pub fn load_config(root: &Path, env_name: &str, argv: &[String]) -> Ini {
    let env_config = root.join("envs").join(env_name).join(format!("{env_name}.ini"));
    if !env_config.exists() {
        crate::fail(&format!("no config for {env_name}: {} not found", env_config.display()));
    }
    let mut ini = Ini::default();
    ini.load(&root.join("vendor/PufferLib/config/default.ini"));
    ini.load(&env_config);
    ini.put("base.env_name", env_name);

    let mut flags = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        if !arg.starts_with("--") {
            crate::fail(&format!("unexpected argument '{arg}'"));
        }
        let arg = arg.trim_start_matches('-');
        let (key, value) = match arg.split_once('=') {
            Some((key, value)) => (key, value.to_string()),
            None if i + 1 < argv.len() && !argv[i + 1].starts_with("--") => {
                i += 1;
                (arg, argv[i].clone())
            }
            None => (arg, "true".to_string()),
        };
        let key = key.replace('-', "_");
        if key == "config" {
            ini.overlay(Path::new(&value));
        } else {
            flags.push((if key.contains('.') { key } else { format!("base.{key}") }, value));
        }
        i += 1;
    }
    for (key, value) in flags {
        ini.put(&key, &value);
    }
    ini
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        assert_eq!(parse("True"), Value::Int(1));
        assert_eq!(parse("20_000_000"), Value::Int(20_000_000));
        assert_eq!(parse("3e7"), Value::Int(30_000_000));
        assert_eq!(parse("0.015"), Value::Float(0.015));
        assert_eq!(parse("3,4,5"), Value::List(vec![3.0, 4.0, 5.0]));
        assert_eq!(parse("None"), Value::Str("None".into()));
        assert_eq!(parse("score "), Value::Str("score ".into()));
    }

    /// A --config file goes over the env's config, and flags over both, wherever they are
    #[test]
    fn overlays() {
        let root = std::env::temp_dir().join(format!("pufferl-config-{}", std::process::id()));
        fs::create_dir_all(root.join("vendor/PufferLib/config")).unwrap();
        fs::create_dir_all(root.join("envs/toy")).unwrap();
        fs::write(root.join("vendor/PufferLib/config/default.ini"), "[base]\nenv_name = None\n[env]\n").unwrap();
        fs::write(root.join("envs/toy/toy.ini"), "[env]\nstar = a\nclock = 900\nrespawn = 0\n").unwrap();
        let variant = root.join("b.ini");
        fs::write(&variant, "[env]\nstar = b\nclock = 1500\n").unwrap();
        let args: Vec<String> = ["--env.clock", "2000", "--config", variant.to_str().unwrap()]
            .iter().map(|arg| arg.to_string()).collect();
        let ini = load_config(&root, "toy", &args);
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(ini.raw("env", "star"), Some("b"));
        assert_eq!(ini.raw("env", "clock"), Some("2000"));
        assert_eq!(ini.raw("env", "respawn"), Some("0"));
        assert_eq!(ini.raw("base", "env_name"), Some("toy"));
    }

    #[test]
    fn comments() {
        assert_eq!(strip_comment("a = 1 # one"), "a = 1 ");
        assert_eq!(strip_comment("a = '#' ; two"), "a = '#' ");
    }
}
