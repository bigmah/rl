//! The dashboard (puf_dashboard_print).

use std::io::{IsTerminal, Write};

use crate::pufferl::{Log, PuffeRL};

const WIDTH: usize = 80;
const BASE_ROWS: usize = 12;
const MAX_USER_ROWS: usize = 15;

pub struct Dashboard {
    tty: bool,
    frame: usize,
    last_size: Option<(usize, usize)>,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Rows and columns of the terminal, or what a terminal that does not say is taken to be.
fn terminal_size() -> (usize, usize) {
    #[cfg(unix)]
    unsafe {
        let mut size: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) == 0 && size.ws_col > 0 && size.ws_row > 0 {
            return (size.ws_row as usize, size.ws_col as usize);
        }
    }
    (1000, WIDTH)
}

pub fn abbrev(mut value: f64) -> String {
    for suffix in ["", "K", "M", "B"] {
        if value < 1000.0 {
            return format!("{value:.1}{suffix}");
        }
        value /= 1000.0;
    }
    format!("{value:.1}T")
}

pub fn duration(seconds: f64) -> String {
    let ms = (seconds.max(0.0) * 1000.0 + 0.5) as u64;
    if ms < 1000 {
        return format!("{ms}ms");
    }
    if ms < 60_000 {
        return format!("{}s {:03}ms", ms / 1000, ms % 1000);
    }
    if ms < 3_600_000 {
        return format!("{}m {:02}s {:03}ms", ms / 60_000, (ms / 1000) % 60, ms % 1000);
    }
    let s = ms / 1000;
    format!("{}d {}h {}m {}s", s / 86_400, (s / 3600) % 24, (s / 60) % 60, s % 60)
}

/// The first `width` characters of `text`, padded on the right to `width`.
fn left(text: &str, width: usize) -> String {
    let cut: String = text.chars().take(width).collect();
    format!("{cut:<width$}")
}

impl Dashboard {
    pub fn new() -> Self {
        Self { tty: std::io::stdout().is_terminal(), frame: 0, last_size: None }
    }

    fn color(&self, code: &'static str) -> &'static str {
        if self.tty { code } else { "" }
    }

    /// Right-aligned in width columns, with unit letters of a number in gray
    fn cell(&self, text: &str, width: usize) -> String {
        let text: String = text.chars().take(width).collect();
        let numeric = text.chars().any(|ch| ch.is_ascii_digit());
        let mut out = String::from(self.color("\x1b[97m"));
        out.push_str(&" ".repeat(width - text.chars().count()));
        for ch in text.chars() {
            let unit = numeric && "%KMBTGdhms".contains(ch);
            out.push_str(self.color(if unit { "\x1b[90m" } else { "\x1b[97m" }));
            out.push(ch);
        }
        out.push_str(self.color("\x1b[0m"));
        out
    }

    pub fn print(&mut self, pufferl: &PuffeRL, log: &Log, epoch: usize) {
        let (a, w, g, r) = (self.color("\x1b[96m"), self.color("\x1b[97m"), self.color("\x1b[90m"), self.color("\x1b[0m"));
        let (rows, cols) = if self.tty { terminal_size() } else { (1000, WIDTH) };
        let env_name = pufferl.ini.text("base", "env_name");
        let steps = log.or("agent_steps", pufferl.global_step as f64);
        let sps = log.or("SPS", 0.0);
        let target = (pufferl.total_epochs * pufferl.batch_size) as f64;
        let remaining = if sps > 0.0 && target > steps { (target - steps) / sps } else { 0.0 };
        let (rollout, train) = (log.or("perf/rollout", 0.0), log.or("perf/train", 0.0));
        let perf_total = rollout + train;

        let mut head = String::new();
        if self.tty {
            head.push_str("\x1b[?2026h");
            head.push_str(if Some((rows, cols)) != self.last_size { "\x1b[H\x1b[J" } else { "\x1b[H" });
            self.last_size = Some((rows, cols));
        }
        let mut stdout = std::io::stdout().lock();
        if self.tty && (cols < WIDTH || rows <= BASE_ROWS) {
            let compact = format!("PufferLib 5.0 ({})  env={env_name}  steps={}  SPS={}  score={:.3}  epoch={epoch}  to_go={}",
                pufferl.gpu.info.backend, abbrev(steps), abbrev(sps), log.or("env/score", 0.0), duration(remaining));
            let compact: String = compact.chars().take(cols.saturating_sub(1)).collect();
            let _ = write!(stdout, "{head}{compact}\x1b[K\x1b[J\x1b[?2026l");
            let _ = stdout.flush();
            return;
        }

        let rule = |l: &str, rt: &str| format!("{w}{l}{}{rt}{r}", "─".repeat(WIDTH - 2));
        let fish_span = 18;
        let fish = (fish_span - 3) - (self.frame % (fish_span - 2));
        self.frame += 1;
        let (used, total) = (log.or("util/vram_used_gb", 0.0), log.or("util/vram_total_gb", 0.0));
        let memory = if total > 0.0 { format!("{used:.1}/{total:.0}G") } else { format!("{used:.1}G") };
        let ram = format!("{:.1}G", log.or("util/cpu_mem_gb", 0.0));

        let mut lines = vec![rule("╭", "╮")];
        // 🐡 is two columns wide
        lines.push(format!("{w}│{r} {a}PufferLib {w}5.0{r}{}{a}🐡{r}{}   {a}GPU{g}:{r}{}    {a}Peak RAM{g}:{r}{}{}{w}│{r}",
            " ".repeat(fish), " ".repeat(fish_span - 2 - fish), self.cell(&memory, 10), self.cell(&ram, 6), " ".repeat(10)));
        lines.push(format!("{w}│{}│{r}", " ".repeat(WIDTH - 2)));

        // Left, its value; middle, its timing (a log key, or seconds given); right, its loss key
        type Row<'a> = (&'a str, String, &'a str, Option<&'a str>, Option<f64>, &'a str, &'a str);
        let mut table: Vec<Row> = vec![
            ("Env", env_name.clone(), "Evaluate", None, Some(rollout), "Losses", "loss/total"),
            ("Params", abbrev(pufferl.num_params() as f64), "  Model", Some("perf/eval_model"), Some(0.0), "policy", "loss/policy"),
            ("Steps", abbrev(steps), "  Env", Some("perf/eval_env"), Some(0.0), "value", "loss/value"),
            ("SPS", abbrev(sps), "  Copy", Some("perf/eval_copy"), Some(0.0), "entropy", "loss/entropy"),
            ("Epoch", epoch.to_string(), "Train", None, Some(train), "old_kl", "loss/old_kl"),
            ("Uptime", duration(log.or("uptime", 0.0)), "  Model", Some("perf/train_model"), Some(0.0), "kl", "loss/kl"),
            ("To go", duration(remaining), "  Misc", Some("perf/train_misc"), Some(0.0), "clipfrac", "loss/clipfrac"),
        ];
        if pufferl.ini.num_opt("train", "target_entropy", 0.0) != 0.0 {
            table.push(("", String::new(), "", None, None, "ent_coef", "train/ent_coef"));
        }
        for (la, value, lb, perf_key, perf_sec, lc, loss_key) in table {
            let sec = match perf_key {
                Some(key) => Some(log.or(key, 0.0)),
                None => perf_sec,
            };
            let timing = match sec {
                Some(sec) => {
                    let percent = if perf_total > 0.0 { (100.0 * sec / perf_total) as i64 } else { 0 };
                    format!("{} {}", self.cell(&duration(sec), 6), self.cell(&format!("{percent}%"), 4))
                }
                None => " ".repeat(11),
            };
            let loss = log.get(loss_key).map(|v| format!("{v:.3}")).unwrap_or_default();
            lines.push(format!("{w}│{r} {a}{}{r} {}    {a}{}{r} {timing}    {a}{}{r} {}    {w}│{r}",
                left(la, 9), self.cell(&value, 13), left(lb, 12), left(lc, 10), self.cell(&loss, 7)));
        }
        lines.push(format!("{w}│{}│{r}", " ".repeat(WIDTH - 2)));

        let user_rows = if self.tty { rows.saturating_sub(BASE_ROWS + 1).min(MAX_USER_ROWS) } else { MAX_USER_ROWS };
        let stats: Vec<(&str, f64)> = log.0.iter()
            .filter(|(key, _)| key.starts_with("env/") && key != "env/n")
            .map(|(key, value)| (&key[4..], *value)).take(2 * user_rows).collect();
        for pair in stats.chunks(2) {
            let stat = |(key, value): &(&str, f64)| format!("{a}{}{r} {w}{value:>9.3}{r}", left(key, 25));
            let right = pair.get(1).map(stat).unwrap_or_else(|| " ".repeat(35));
            lines.push(format!("{w}│{r} {}   {right}    {w}│{r}", stat(&pair[0])));
        }
        lines.push(rule("╰", "╯"));

        let eol = if self.tty { "\x1b[K\n" } else { "\n" };
        let mut out = head;
        for line in &lines {
            out.push_str(line);
            out.push_str(eol);
        }
        if self.tty {
            out.push_str("\x1b[J\x1b[?2026l");
        }
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}
