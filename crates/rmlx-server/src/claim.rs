//! Metal claim — one MLX process per Mac.
//!
//! The Apple Silicon Metal context is exclusive per process. A GPU command
//! holds an exclusive `flock` on one machine-wide file, [`CLAIM_PATH`], from
//! its first GPU call to its exit. The flock is the only gate. The kernel
//! releases it when the holder exits or is killed, so the next process gets
//! the claim with no operator action. The file stays on disk: a process that
//! deleted it would let the next process lock a new file while the old holder
//! still runs.
//!
//! The file body is `<pid> <argv>` of the last holder. It only gives the
//! refusal message its holder. A body is never a reason to claim or to refuse.
//!
//! CPU-only runs (`--device cpu`) take no claim.
//!
//! `flock(2)` is advisory: it stops two rMLX processes, not Python
//! `mlx_lm.server` or ollama.

use std::fs::{File, OpenOptions, Permissions, TryLockError};
use std::io::{self, Read as _, Seek as _, SeekFrom, Write as _};
use std::os::fd::{AsFd, BorrowedFd, IntoRawFd as _, OwnedFd};
use std::os::unix::fs::{FileExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The one claim file. `/var/tmp` is machine-wide, and macOS `tmp_cleaner`
/// does not clean it (it cleans `/tmp`). `std::env::temp_dir()` is per user.
const CLAIM_PATH: &str = "/var/tmp/rmlx.claim";

/// Separate coordination file used only by the experimental budget mode.
const BUDGET_PATH: &str = "/var/tmp/rmlx.budget.json";

/// Leave this fraction of physical RAM outside all rMLX budget reservations.
const OS_HEADROOM_PERCENT: u128 = 15;
const PERCENT_DENOMINATOR: u128 = 100;
const BUDGET_COORDINATION_WAIT: Duration = Duration::from_millis(250);
const BYTES_PER_GB: u64 = 1024 * 1024 * 1024;

/// Builds before the machine-wide claim hold `/tmp/rmlx.<port>.claim`. The
/// probe of these files is for one release: remove it in the release after
/// next.
const LEGACY_CLAIM_DIR: &str = "/tmp";
const LEGACY_CLAIM_PREFIX: &str = "rmlx.";
const LEGACY_CLAIM_SUFFIX: &str = ".claim";

/// `O_NONBLOCK`: the open of a FIFO planted at a claim path must not block.
const OPEN_FLAGS: libc::c_int = libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;

/// Error returned when `try_claim` cannot get the claim.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    /// Another process holds the claim.
    #[error("{}", refusal_message(*.holder_pid, .holder_command, .path))]
    AlreadyHeld {
        /// PID the holder wrote into the file body. It can be stale or
        /// forged. It is `None` when the holder wrote no body, and while a
        /// holder is starting the body can be empty or still the previous
        /// holder's: the holder truncates and rewrites it after it locks.
        holder_pid: Option<u32>,
        /// Command line of the holder, as it recorded it.
        holder_command: String,
        /// The lock file the holder has locked.
        path: PathBuf,
    },

    /// OS error on the claim file, or the path is not a regular file with
    /// one link.
    #[error("Metal claim I/O error at {}: {source}", .path.display())]
    Io {
        /// The file the claim operation used.
        path: PathBuf,
        /// Underlying I/O error from the OS.
        #[source]
        source: io::Error,
    },

    /// The requested reservation would exceed the machine-wide memory budget.
    #[error(
        "experimental memory budget denied: requested {requested_gb:.3} GiB with {allocated_gb:.3} GiB already reserved; machine budget is {limit_gb:.3} GiB"
    )]
    BudgetExceeded {
        /// New reservation requested by this process.
        requested_gb: f64,
        /// Sum of existing live reservations.
        allocated_gb: f64,
        /// Available budget after OS headroom.
        limit_gb: f64,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct BudgetLedger {
    available_gb: f64,
    entries: Vec<BudgetEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BudgetEntry {
    id: String,
    pid: u32,
    requested_gb: f64,
    argv: Vec<String>,
}

impl Default for BudgetLedger {
    fn default() -> Self {
        Self {
            available_gb: 0.0,
            entries: Vec::new(),
        }
    }
}

/// Holds the Metal claim until it is dropped. The flock releases when the
/// last fd on its open file closes; the file stays. A child that inherits the
/// fd holds the flock too.
#[derive(Debug)]
pub struct MetalClaim {
    file: File,
}

/// A machine-wide memory reservation held by the experimental multi-process
/// mode. Its ledger entry is removed on drop; a later claimant removes it if
/// the process was killed before Drop could run.
#[derive(Debug)]
pub struct MemoryBudgetClaim {
    ledger_path: PathBuf,
    coordination_path: Option<PathBuf>,
    id: String,
    cleanup_on_drop: AtomicBool,
}

impl MemoryBudgetClaim {
    /// Keep this reservation in the ledger until this process exits.
    ///
    /// This is used when a server shutdown deadline expires while a blocking
    /// GPU job can still be running. The next ledger operation reaps the
    /// entry after the kernel reports this PID gone.
    pub fn hold_until_process_exit(&self) {
        self.cleanup_on_drop.store(false, Ordering::Release);
    }
}

impl Drop for MemoryBudgetClaim {
    fn drop(&mut self) {
        if self.cleanup_on_drop.load(Ordering::Acquire) {
            if let Err(error) = remove_budget_entry(
                self.coordination_path.as_deref(),
                &self.ledger_path,
                &self.id,
            ) {
                tracing::warn!(error = %error, "could not release experimental memory reservation; it will be reaped after process exit");
            }
        }
    }
}

/// GPU coordination held by a CLI operation: the default exclusive Metal
/// claim or an experimental memory-budget reservation.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuClaim {
    /// Existing exclusive flock claim.
    Metal(MetalClaim),
    /// Experimental ledger reservation.
    Budget(MemoryBudgetClaim),
}

impl GpuClaim {
    /// Preserve the GPU reservation beyond its Rust guard when work may still
    /// run after the server shutdown deadline.
    pub fn hold_until_process_exit(&self) -> io::Result<()> {
        match self {
            Self::Metal(claim) => claim.hold_until_process_exit(),
            Self::Budget(claim) => {
                claim.hold_until_process_exit();
                Ok(())
            }
        }
    }
}

impl MetalClaim {
    /// Keep the claim past this guard, until the process is gone. For an exit
    /// that leaves a job on the GPU: the next process must not get the claim
    /// while this one still runs.
    ///
    /// # Errors
    /// The OS could not duplicate the fd; the claim then ends with the guard.
    pub fn hold_until_process_exit(&self) -> io::Result<()> {
        // The flock belongs to the open file, so a second fd that is never
        // closed keeps it.
        let _never_closed = self.file.try_clone()?.into_raw_fd();
        Ok(())
    }
}

impl AsFd for MetalClaim {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

/// Give up the claim's fd, for a process that hands the claim to a child it
/// starts.
impl From<MetalClaim> for OwnedFd {
    fn from(claim: MetalClaim) -> Self {
        claim.file.into()
    }
}

/// Get the machine-wide Metal claim, or refuse if another process holds it.
///
/// For one release this also refuses when a build before the machine-wide
/// claim holds a `/tmp/rmlx.<port>.claim`.
///
/// # Errors
/// [`ClaimError::AlreadyHeld`] names the holder. [`ClaimError::Io`] is an OS
/// error; the caller must not use the GPU then either.
pub fn try_claim() -> Result<MetalClaim, ClaimError> {
    refuse_held_legacy_claim(Path::new(LEGACY_CLAIM_DIR))?;
    claim_default_at(Path::new(CLAIM_PATH), Path::new(BUDGET_PATH))
}

/// Reserve `requested_gb` from total physical memory minus the fixed OS
/// headroom. This entry point is only used when the experimental CLI flag is
/// explicitly set; [`try_claim`] remains the unchanged default path.
///
/// # Errors
/// Returns [`ClaimError::BudgetExceeded`] when the reservation does not fit,
/// or [`ClaimError::Io`] when the ledger or system memory probe fails.
pub fn try_claim_with_budget(requested_gb: f64) -> Result<MemoryBudgetClaim, ClaimError> {
    refuse_held_legacy_claim(Path::new(LEGACY_CLAIM_DIR))?;
    let total_bytes = rmlx_core::system::physical_memory_bytes()
        .map_err(|source| io_error(Path::new(BUDGET_PATH), source))?;
    reserve_budget_coordinated_at(
        Path::new(CLAIM_PATH),
        Path::new(BUDGET_PATH),
        requested_gb,
        total_bytes,
    )
}

fn claim_default_at(claim_path: &Path, ledger_path: &Path) -> Result<MetalClaim, ClaimError> {
    let claim = claim_at(claim_path)?;
    refuse_if_budget_reserved(ledger_path, claim_path)?;
    Ok(claim)
}

fn reserve_budget_coordinated_at(
    claim_path: &Path,
    ledger_path: &Path,
    requested_gb: f64,
    total_bytes: u64,
) -> Result<MemoryBudgetClaim, ClaimError> {
    let coordination = acquire_budget_coordination_lock(claim_path)?;
    let claim = reserve_budget_at(ledger_path, requested_gb, total_bytes).map(|mut claim| {
        claim.coordination_path = Some(claim_path.to_path_buf());
        claim
    });
    drop(coordination);
    claim
}

fn refuse_if_budget_reserved(ledger_path: &Path, claim_path: &Path) -> Result<(), ClaimError> {
    let file = match open_checked(OpenOptions::new().read(true).write(true), ledger_path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(io_error(ledger_path, source)),
    };
    file.lock()
        .map_err(|source| io_error(ledger_path, source))?;
    let ledger = read_budget_ledger(&file, ledger_path, false)?;
    let mut ledger = ledger;
    reap_orphans(&mut ledger);
    if let Some(holder) = ledger.entries.first() {
        let command = holder.argv.join(" ");
        return Err(ClaimError::AlreadyHeld {
            holder_pid: Some(holder.pid),
            holder_command: command,
            path: claim_path.to_path_buf(),
        });
    }
    Ok(())
}

fn reserve_budget_at(
    path: &Path,
    requested_gb: f64,
    total_bytes: u64,
) -> Result<MemoryBudgetClaim, ClaimError> {
    if !requested_gb.is_finite() || requested_gb <= 0.0 {
        return Err(io_error(
            path,
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "requested memory budget must be a finite positive number",
            ),
        ));
    }
    let limit_bytes =
        u128::from(total_bytes) * (PERCENT_DENOMINATOR - OS_HEADROOM_PERCENT) / PERCENT_DENOMINATOR;
    let limit_gb = limit_bytes as f64 / BYTES_PER_GB as f64;
    let requested_bytes = gb_to_bytes_ceil(requested_gb).ok_or_else(|| {
        io_error(
            path,
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "requested memory budget is too large to represent",
            ),
        )
    })?;
    let (file, created) = open_budget_file(path).map_err(|source| io_error(path, source))?;
    file.lock().map_err(|source| io_error(path, source))?;
    let mut ledger = read_budget_ledger(&file, path, created)?;
    let entry_count_before_reap = ledger.entries.len();
    reap_orphans(&mut ledger);
    ledger.available_gb = limit_gb;
    let allocated_bytes = ledger.entries.iter().try_fold(0_u128, |sum, entry| {
        gb_to_bytes_ceil(entry.requested_gb).and_then(|bytes| sum.checked_add(u128::from(bytes)))
    });
    let Some(allocated_bytes) = allocated_bytes else {
        return Err(io_error(
            path,
            io::Error::new(
                io::ErrorKind::InvalidData,
                "ledger reservation total overflowed",
            ),
        ));
    };
    let allocated_gb = allocated_bytes as f64 / BYTES_PER_GB as f64;
    let exceeds_budget = allocated_bytes
        .checked_add(u128::from(requested_bytes))
        .is_none_or(|total| total > limit_bytes);
    if exceeds_budget {
        if ledger.entries.len() != entry_count_before_reap {
            write_budget_ledger(path, &ledger)?;
        }
        return Err(ClaimError::BudgetExceeded {
            requested_gb,
            allocated_gb,
            limit_gb,
        });
    }

    let id = uuid::Uuid::new_v4().to_string();
    ledger.entries.push(BudgetEntry {
        id: id.clone(),
        pid: std::process::id(),
        requested_gb,
        argv: std::env::args_os()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
    });
    write_budget_ledger(path, &ledger)?;
    tracing::info!(
        pid = std::process::id(),
        requested_gb,
        allocated_gb,
        limit_gb,
        "experimental memory budget reserved"
    );
    drop(file);
    Ok(MemoryBudgetClaim {
        ledger_path: path.to_path_buf(),
        coordination_path: None,
        id,
        cleanup_on_drop: AtomicBool::new(true),
    })
}

fn open_budget_file(path: &Path) -> io::Result<(File, bool)> {
    match open_checked(
        OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o666),
        path,
    ) {
        Ok(file) => {
            file.set_permissions(Permissions::from_mode(0o666))?;
            Ok((file, true))
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            open_checked(OpenOptions::new().read(true).write(true), path).map(|file| (file, false))
        }
        Err(error) => Err(error),
    }
}

fn read_budget_ledger(
    file: &File,
    path: &Path,
    allow_new_empty_file: bool,
) -> Result<BudgetLedger, ClaimError> {
    let mut contents = String::new();
    let mut reader = file;
    reader
        .seek(SeekFrom::Start(0))
        .map(|_| ())
        .and_then(|()| reader.read_to_string(&mut contents))
        .map_err(|source| io_error(path, source))?;
    if contents.trim().is_empty() {
        if allow_new_empty_file {
            return Ok(BudgetLedger::default());
        }
        return Err(io_error(
            path,
            io::Error::new(
                io::ErrorKind::InvalidData,
                "existing budget ledger is empty",
            ),
        ));
    }
    let ledger: BudgetLedger = serde_json::from_str(&contents)
        .map_err(|source| io_error(path, io::Error::new(io::ErrorKind::InvalidData, source)))?;
    validate_budget_ledger(&ledger).map_err(|source| io_error(path, source))?;
    Ok(ledger)
}

fn validate_budget_ledger(ledger: &BudgetLedger) -> io::Result<()> {
    if !ledger.available_gb.is_finite() || ledger.available_gb < 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ledger available_gb must be finite and non-negative",
        ));
    }
    for (index, entry) in ledger.entries.iter().enumerate() {
        if entry.id.is_empty()
            || entry.pid == 0
            || !entry.requested_gb.is_finite()
            || entry.requested_gb <= 0.0
            || gb_to_bytes_ceil(entry.requested_gb).is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ledger entry has an invalid id, PID, or requested_gb",
            ));
        }
        if ledger
            .entries
            .iter()
            .take(index)
            .any(|previous| previous.id == entry.id)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ledger entry ids must be unique",
            ));
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "positive finite byte counts are range-checked and rounded up before conversion"
)]
fn gb_to_bytes_ceil(gb: f64) -> Option<u64> {
    let bytes = (gb * BYTES_PER_GB as f64).ceil();
    if !bytes.is_finite() || bytes <= 0.0 || bytes >= u64::MAX as f64 {
        return None;
    }
    Some(bytes as u64)
}

struct LedgerTempFile {
    path: PathBuf,
    published: bool,
}

impl Drop for LedgerTempFile {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn write_budget_ledger(path: &Path, ledger: &BudgetLedger) -> Result<(), ClaimError> {
    write_budget_ledger_with_hook(path, ledger, || Ok(()))
}

fn write_budget_ledger_with_hook(
    path: &Path,
    ledger: &BudgetLedger,
    before_publish: impl FnOnce() -> io::Result<()>,
) -> Result<(), ClaimError> {
    let bytes = serde_json::to_vec_pretty(ledger)
        .map_err(|source| io_error(path, io::Error::new(io::ErrorKind::InvalidData, source)))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| io_error(path, io::Error::other("ledger path has no file name")))?;
    let temporary_path = parent.join(format!(
        ".{}.tmp.{}.{}",
        file_name.to_string_lossy(),
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true).mode(0o666);
    let mut temporary = open_checked(&mut options, &temporary_path)
        .map_err(|source| io_error(&temporary_path, source))?;
    let mut cleanup = LedgerTempFile {
        path: temporary_path.clone(),
        published: false,
    };
    temporary
        .set_permissions(Permissions::from_mode(0o666))
        .and_then(|()| temporary.write_all(&bytes))
        .and_then(|()| temporary.sync_all())
        .map_err(|source| io_error(&temporary_path, source))?;
    before_publish().map_err(|source| io_error(path, source))?;
    std::fs::rename(&temporary_path, path).map_err(|source| io_error(path, source))?;
    cleanup.published = true;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error(parent, source))
}

fn reap_orphans(ledger: &mut BudgetLedger) {
    ledger.entries.retain(|entry| {
        match rmlx_core::system::process_exists(entry.pid) {
            Ok(alive) => alive,
            Err(error) => {
                tracing::warn!(pid = entry.pid, error = %error, "cannot determine whether memory-budget entry is orphaned; retaining it conservatively");
                true
            }
        }
    });
}

fn acquire_coordination_lock(path: &Path) -> io::Result<File> {
    let (file, _writable) = open_claim_file(path)?;
    file.lock()?;
    Ok(file)
}

fn acquire_budget_coordination_lock(path: &Path) -> Result<File, ClaimError> {
    let (file, _writable) = open_claim_file(path).map_err(|source| io_error(path, source))?;
    let started = Instant::now();
    loop {
        if locked(file.try_lock()).map_err(|source| io_error(path, source))? {
            return Ok(file);
        }
        if started.elapsed() >= BUDGET_COORDINATION_WAIT {
            let (holder_pid, holder_command) = read_holder(&file);
            return Err(ClaimError::AlreadyHeld {
                holder_pid,
                holder_command,
                path: path.to_path_buf(),
            });
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn remove_budget_entry(coordination_path: Option<&Path>, path: &Path, id: &str) -> io::Result<()> {
    let _coordination = coordination_path
        .map(acquire_coordination_lock)
        .transpose()?;
    let file = match open_checked(OpenOptions::new().read(true).write(true), path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    file.lock()?;
    let mut ledger = read_budget_ledger(&file, path, false).map_err(claim_error_to_io)?;
    let old_len = ledger.entries.len();
    ledger.entries.retain(|entry| entry.id != id);
    if ledger.entries.len() != old_len {
        let total_bytes = rmlx_core::system::physical_memory_bytes()?;
        ledger.available_gb = (u128::from(total_bytes)
            * (PERCENT_DENOMINATOR - OS_HEADROOM_PERCENT)
            / PERCENT_DENOMINATOR) as f64
            / BYTES_PER_GB as f64;
        write_budget_ledger(path, &ledger).map_err(claim_error_to_io)?;
    }
    Ok(())
}

fn claim_error_to_io(error: ClaimError) -> io::Error {
    match error {
        ClaimError::Io { source, .. } => source,
        other @ (ClaimError::AlreadyHeld { .. } | ClaimError::BudgetExceeded { .. }) => {
            io::Error::other(other.to_string())
        }
    }
}

/// Report whether a process holds the Metal claim, without taking it.
/// `Ok(())` means no process holds it. The probe writes no body and creates
/// no file. It holds a shared lock for a moment, and a GPU command that
/// starts in that moment is refused.
///
/// # Errors
/// [`ClaimError::AlreadyHeld`] names the holder. [`ClaimError::Io`] is an OS
/// error.
pub fn probe_claim() -> Result<(), ClaimError> {
    probe_in_with_ledger(
        Path::new(LEGACY_CLAIM_DIR),
        Path::new(CLAIM_PATH),
        Some(Path::new(BUDGET_PATH)),
    )
}

#[cfg(test)]
fn claim_in(legacy_dir: &Path, path: &Path) -> Result<MetalClaim, ClaimError> {
    refuse_held_legacy_claim(legacy_dir)?;
    claim_at(path)
}

#[cfg(test)]
fn probe_in(legacy_dir: &Path, path: &Path) -> Result<(), ClaimError> {
    probe_in_with_ledger(legacy_dir, path, None)
}

fn probe_in_with_ledger(
    legacy_dir: &Path,
    path: &Path,
    ledger_path: Option<&Path>,
) -> Result<(), ClaimError> {
    refuse_held_legacy_claim(legacy_dir)?;
    let file = match open_checked(OpenOptions::new().read(true), path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(io_error(path, source)),
    };
    refuse_if_held(&file, path, None)?;
    if let Some(ledger_path) = ledger_path {
        refuse_if_budget_reserved(ledger_path, path)?;
    }
    Ok(())
}

fn claim_at(path: &Path) -> Result<MetalClaim, ClaimError> {
    let (file, writable) = open_claim_file(path).map_err(|e| io_error(path, e))?;
    if !locked(file.try_lock()).map_err(|e| io_error(path, e))? {
        let (holder_pid, holder_command) = read_holder(&file);
        return Err(ClaimError::AlreadyHeld {
            holder_pid,
            holder_command,
            path: path.to_path_buf(),
        });
    }
    if writable {
        write_holder(&file).map_err(|e| io_error(path, e))?;
    }
    tracing::info!(pid = std::process::id(), path = %path.display(), writable, "Metal claim acquired");
    Ok(MetalClaim { file })
}

fn io_error(path: &Path, source: io::Error) -> ClaimError {
    ClaimError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Open the claim file for the lock. Returns the file and whether it is
/// writable. A file this call creates gets mode 0666, so every user can open
/// it later. When another user's file refuses a write open, a read-only open
/// still takes the flock.
fn open_claim_file(path: &Path) -> io::Result<(File, bool)> {
    let open = |write: bool, create: bool| {
        open_checked(
            OpenOptions::new()
                .read(true)
                .write(write)
                .create_new(create)
                .mode(0o666),
            path,
        )
    };
    match open(true, true) {
        Ok(file) => {
            // The umask masks the create mode.
            file.set_permissions(Permissions::from_mode(0o666))?;
            Ok((file, true))
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => match open(true, false) {
            Ok(file) => Ok((file, true)),
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                Ok((open(false, false)?, false))
            }
            Err(e) => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// Open `path` following no symlink, and refuse anything but a regular file
/// with one link: a hard link planted at the path would make the body write
/// land in its target.
fn open_checked(options: &mut OpenOptions, path: &Path) -> io::Result<File> {
    let file = open_no_follow(options, path)?;
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Err(io::Error::other("the claim path is not a regular file"));
    }
    if meta.nlink() != 1 {
        return Err(io::Error::other("the claim file has more than one link"));
    }
    Ok(file)
}

fn open_no_follow(options: &mut OpenOptions, path: &Path) -> io::Result<File> {
    options.custom_flags(OPEN_FLAGS).open(path)
}

/// `Ok(false)` when another open file holds a conflicting lock.
fn locked(result: Result<(), TryLockError>) -> io::Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// Refuse when another open file holds an exclusive lock on `file`. The
/// shared lock taken here is released when `file` closes.
fn refuse_if_held(file: &File, path: &Path, command: Option<&str>) -> Result<(), ClaimError> {
    if locked(file.try_lock_shared()).map_err(|e| io_error(path, e))? {
        return Ok(());
    }
    let (holder_pid, recorded_command) = read_holder(file);
    Err(ClaimError::AlreadyHeld {
        holder_pid,
        holder_command: command.map_or(recorded_command, str::to_owned),
        path: path.to_path_buf(),
    })
}

fn write_holder(file: &File) -> io::Result<()> {
    let argv: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let body = format!("{} {}", std::process::id(), argv.join(" "));
    file.set_len(0)?;
    file.write_all_at(body.as_bytes(), 0)
}

/// Parse a body `<pid> <argv>`, or the bare `<pid>` a legacy claim holds.
fn read_holder(mut file: &File) -> (Option<u32>, String) {
    let mut body = String::new();
    if file.read_to_string(&mut body).is_err() {
        return (None, String::new());
    }
    let body = body.trim();
    let (pid, command) = body.split_once(' ').unwrap_or((body, ""));
    (pid.parse().ok(), command.to_owned())
}

fn refusal_message(holder_pid: Option<u32>, holder_command: &str, path: &Path) -> String {
    let command = if holder_command.is_empty() {
        "command not recorded"
    } else {
        holder_command
    };
    let recorded = match holder_pid {
        Some(pid) => format!("The holder recorded PID {pid} ({command}); the record can be stale."),
        None => format!("The holder recorded no PID ({command})."),
    };
    format!(
        "the Metal claim is held. {recorded} Find the process that holds the lock with \
         `lsof {}` (as root to see another user's process), stop it with `kill <PID>`, \
         then run this command again.",
        path.display()
    )
}

/// Refuse when a legacy per-port claim in `dir` is held. The probe takes a
/// shared lock for a moment and changes no file. Old builds made only regular
/// files, so it skips an entry that vanished, is a symlink or is not a regular
/// file; any other open error refuses. A hard-linked file keeps its inode's
/// flock and is probed.
fn refuse_held_legacy_claim(dir: &Path) -> Result<(), ClaimError> {
    let entries = std::fs::read_dir(dir).map_err(|e| io_error(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_error(dir, e))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(LEGACY_CLAIM_PREFIX) && name.ends_with(LEGACY_CLAIM_SUFFIX)) {
            continue;
        }
        let path = entry.path();
        let file = match open_no_follow(OpenOptions::new().read(true), &path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => continue,
            Err(source) => return Err(io_error(&path, source)),
        };
        if !file
            .metadata()
            .map_err(|e| io_error(&path, e))?
            .file_type()
            .is_file()
        {
            continue;
        }
        refuse_if_held(
            &file,
            &path,
            Some("an rmlx build older than the machine-wide claim"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "claim_tests.rs"]
mod tests;
