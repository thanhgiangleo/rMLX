use super::*;
use std::io::{BufRead as _, BufReader, IsTerminal as _};
use std::os::unix::fs::symlink;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// When `Command::spawn` forks, the child holds a copy of every open fd of
/// this process from the fork until its exec closes the `O_CLOEXEC` ones; a
/// flock this process drops in that window stays held by the copy. A test
/// that needs its own claim released at drop holds this lock, and so does
/// every spawn.
static SPAWN: Mutex<()> = Mutex::new(());

fn no_spawn() -> MutexGuard<'static, ()> {
    SPAWN.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lock_in(dir: &TempDir) -> PathBuf {
    dir.path().join("lock")
}

fn is_held(result: &Result<MetalClaim, ClaimError>) -> bool {
    matches!(result, Err(ClaimError::AlreadyHeld { .. }))
}

fn is_io<T>(result: &Result<T, ClaimError>) -> bool {
    matches!(result, Err(ClaimError::Io { .. }))
}

fn make_fifo(path: &Path) {
    let _spawn = no_spawn();
    let status = Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo failed: {status}");
}

/// The body `write_holder` writes for this process.
fn own_body() -> String {
    let argv: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    format!("{} {}", std::process::id(), argv.join(" "))
}

/// A child process that holds the claim on a lock file. The child is this
/// test binary, run as `claim_holder_child`.
struct Holder {
    child: Child,
    stdin: Option<ChildStdin>,
}

impl Holder {
    fn start(lock: &Path) -> Self {
        let spawn = no_spawn();
        let mut child = Command::new(std::env::current_exe().expect("current_exe"))
            .args([
                "claim::tests::claim_holder_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the holder child");
        drop(spawn);
        let mut stdin = child.stdin.take().expect("child stdin");
        writeln!(stdin, "{}", lock.display()).expect("send the lock path");
        let stdout = BufReader::new(child.stdout.take().expect("child stdout"));
        let held = stdout
            .lines()
            .map_while(Result::ok)
            .any(|line| line.ends_with(&format!(" held {}", child.id())));
        assert!(held, "the child did not report that it holds the claim");
        Holder {
            child,
            stdin: Some(stdin),
        }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Close the child's stdin: the child calls `process::exit` with the
    /// claim still open, so no `Drop` runs.
    fn exit(mut self) {
        drop(self.stdin.take());
        let status = self.child.wait().expect("wait for the child");
        assert!(status.success(), "the child failed: {status}");
    }

    fn kill(mut self) {
        self.child.kill().expect("SIGKILL the child");
        self.child.wait().expect("wait for the child");
    }
}

/// The child half of the two-process tests. It reads a lock path on stdin,
/// holds the claim on it, prints `held <pid>`, and exits when stdin closes.
#[test]
#[ignore = "child process of the two-process claim tests; the parent test starts it"]
fn claim_holder_child() {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        return;
    }
    let mut path = String::new();
    if stdin.lock().read_line(&mut path).is_err() || path.trim().is_empty() {
        return;
    }
    let _claim = claim_at(Path::new(path.trim())).expect("the child must get the claim");
    println!("held {}", std::process::id());
    io::stdout().flush().expect("flush stdout");
    let mut rest = Vec::new();
    let _ = stdin.lock().read_to_end(&mut rest);
    std::process::exit(0);
}

#[test]
fn production_claim_path_is_one_file_under_var_tmp() {
    let path = Path::new(CLAIM_PATH);
    assert_eq!(path.parent(), Some(Path::new("/var/tmp")));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name");
    assert!(
        !name.contains(|c: char| c.is_ascii_digit()),
        "the claim file name must not carry a port: {name}"
    );
}

/// flock locks belong to the open file, so a second open in this process
/// conflicts. A per-process lock (`fcntl`, `lockf`) would not.
#[test]
fn second_claim_in_same_process_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let first = claim_at(&lock).expect("first claim");
    let second = claim_at(&lock);
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = second else {
        panic!("the second claim must be refused, got {second:?}");
    };
    assert_eq!(holder_pid, Some(std::process::id()));
    drop(first);
}

#[test]
fn drop_keeps_the_file_and_the_next_claim_locks_it() {
    let _no_spawn = no_spawn();
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    drop(claim_at(&lock).expect("first claim"));
    assert!(lock.exists(), "the claim file must stay after drop");
    let next = claim_at(&lock).expect("the next claim after drop");
    assert!(
        is_held(&claim_at(&lock)),
        "the next claim must hold the lock"
    );
    drop(next);
}

#[test]
fn symlink_at_claim_path_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let target = dir.path().join("target");
    std::fs::write(&target, "keep").expect("write target");
    let lock = lock_in(&dir);
    symlink(&target, &lock).expect("plant a symlink");
    let result = claim_at(&lock);
    assert!(
        matches!(result, Err(ClaimError::Io { .. })),
        "a symlink must be refused, got {result:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&target).expect("read target"),
        "keep"
    );
}

#[test]
fn claim_file_mode_is_0666() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let _claim = claim_at(&lock).expect("claim");
    let mode = std::fs::metadata(&lock).expect("stat").permissions().mode();
    assert_eq!(mode & 0o777, 0o666);
}

/// A file this user cannot write, as another user's claim file is: the claim
/// takes the flock through a read-only open and writes no body.
#[test]
fn read_only_file_is_still_locked() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    std::fs::write(&lock, "").expect("create the file");
    std::fs::set_permissions(&lock, Permissions::from_mode(0o444)).expect("chmod 0444");
    assert!(
        OpenOptions::new().write(true).open(&lock).is_err(),
        "this test needs a user that a 0444 file refuses a write open"
    );
    let claim = claim_at(&lock).expect("a read-only file must still be claimed");
    assert!(
        is_held(&claim_at(&lock)),
        "the read-only claim must hold the lock"
    );
    assert_eq!(std::fs::read_to_string(&lock).expect("read body"), "");
    drop(claim);
}

/// The last holder's PID stays in the body, and the OS can give that PID to
/// another process. Only the flock decides.
#[test]
fn unlocked_file_with_live_pid_body_is_claimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let parent = std::os::unix::process::parent_id();
    assert_ne!(parent, std::process::id());
    std::fs::write(&lock, format!("{parent} some command")).expect("write body");
    let claim = claim_at(&lock).expect("an unlocked file must be claimed");
    let body = std::fs::read_to_string(&lock).expect("read body");
    let (pid, _) = body.split_once(' ').expect("body is `<pid> <argv>`");
    assert_eq!(pid, std::process::id().to_string());
    drop(claim);
}

#[test]
fn new_body_replaces_a_longer_one_whole() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    std::fs::write(&lock, format!("4242 {}", "x".repeat(4096))).expect("write a long body");
    let claim = claim_at(&lock).expect("claim");
    assert_eq!(
        std::fs::read_to_string(&lock).expect("read body"),
        own_body()
    );
    drop(claim);
}

/// A hard link planted at the claim path would make the body write land in
/// the link's target.
#[test]
fn hard_link_at_claim_path_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let target = dir.path().join("target");
    std::fs::write(&target, "keep").expect("write target");
    let lock = lock_in(&dir);
    std::fs::hard_link(&target, &lock).expect("plant a hard link");
    let result = claim_at(&lock);
    assert!(
        is_io(&result),
        "a hard link must be refused, got {result:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&target).expect("read target"),
        "keep"
    );
}

/// Without the regular-file check a FIFO still fails, at the flock, and a
/// directory still fails, at the link count; only the reason tells the check
/// ran.
fn assert_not_regular<T: std::fmt::Debug>(result: &Result<T, ClaimError>) {
    let Err(ClaimError::Io { source, .. }) = result else {
        panic!("a path that is not a regular file must be refused, got {result:?}");
    };
    assert!(
        source.to_string().contains("not a regular file"),
        "refused for another reason: {source}"
    );
}

#[test]
fn fifo_at_claim_path_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    make_fifo(&lock);
    assert_not_regular(&claim_at(&lock));
}

#[test]
fn directory_at_claim_path_is_refused_by_the_probe() {
    let legacy = TempDir::new().expect("legacy dir");
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    std::fs::create_dir(&lock).expect("plant a directory");
    assert_not_regular(&probe_in(legacy.path(), &lock));
}

#[test]
fn refusal_message_names_the_record_and_how_to_confirm_it() {
    let message = ClaimError::AlreadyHeld {
        holder_pid: Some(4242),
        holder_command: "rmlx serve --port 8080".to_owned(),
        path: PathBuf::from(CLAIM_PATH),
    }
    .to_string();
    assert!(message.contains("recorded PID 4242"), "{message}");
    assert!(message.contains("rmlx serve --port 8080"), "{message}");
    assert!(message.contains(&format!("lsof {CLAIM_PATH}")), "{message}");
    assert!(message.contains("kill <PID>"), "{message}");
    assert!(
        !message.contains("kill 4242"),
        "the recorded PID is not verified: {message}"
    );
    assert_no_delete_word(&message);
}

#[test]
fn refusal_message_without_a_pid_promises_none() {
    let message = ClaimError::AlreadyHeld {
        holder_pid: None,
        holder_command: String::new(),
        path: PathBuf::from(CLAIM_PATH),
    }
    .to_string();
    assert!(message.contains("recorded no PID"), "{message}");
    assert!(message.contains(&format!("lsof {CLAIM_PATH}")), "{message}");
    let lower = message.to_lowercase();
    for promise in ["yet", "in a moment", "soon"] {
        assert!(!lower.contains(promise), "{message}");
    }
    assert_no_delete_word(&message);
}

fn assert_no_delete_word(message: &str) {
    let lower = message.to_lowercase();
    for word in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        assert!(
            !["rm", "unlink", "delete", "remove"].contains(&word),
            "the refusal must not tell the operator to delete a file: {message}"
        );
    }
}

#[test]
fn second_process_is_refused_and_named() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    let result = claim_at(&lock);
    let Err(ClaimError::AlreadyHeld {
        holder_pid,
        holder_command,
        path,
    }) = result
    else {
        panic!("a claim held by another process must be refused, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert_eq!(path, lock);
    assert!(
        holder_command.contains("claim_holder_child"),
        "the refusal must name the holder's command: {holder_command}"
    );
    holder.exit();
}

#[test]
fn exited_holder_is_reclaimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    assert!(is_held(&claim_at(&lock)));
    holder.exit();
    claim_at(&lock).expect("the claim of an exited holder must be free");
}

#[test]
fn killed_holder_is_reclaimed() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    assert!(is_held(&claim_at(&lock)));
    holder.kill();
    claim_at(&lock).expect("the claim of a killed holder must be free");
}

fn legacy_claim_in(dir: &TempDir) -> PathBuf {
    dir.path()
        .join(format!("{LEGACY_CLAIM_PREFIX}8080{LEGACY_CLAIM_SUFFIX}"))
}

#[test]
fn held_legacy_claim_is_refused_and_kept() {
    let dir = TempDir::new().expect("temp dir");
    let legacy = legacy_claim_in(&dir);
    let holder = Holder::start(&legacy);
    let result = refuse_held_legacy_claim(dir.path());
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("a held legacy claim must be refused, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert!(legacy.exists(), "the probe must keep the legacy file");
    holder.exit();
    assert!(legacy.exists(), "the probe must keep the legacy file");
}

#[test]
fn free_legacy_claim_is_passed_and_kept() {
    let dir = TempDir::new().expect("temp dir");
    let legacy = legacy_claim_in(&dir);
    std::fs::write(&legacy, "4242").expect("write a legacy body");
    refuse_held_legacy_claim(dir.path()).expect("a free legacy claim must pass");
    assert_eq!(std::fs::read_to_string(&legacy).expect("read body"), "4242");
}

#[test]
fn claim_refuses_while_a_legacy_claim_is_held() {
    let legacy_dir = TempDir::new().expect("legacy dir");
    let holder = Holder::start(&legacy_claim_in(&legacy_dir));
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let result = claim_in(legacy_dir.path(), &lock);
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("a held legacy claim must refuse the claim, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert!(!lock.exists(), "a refused claim must not create the file");
    holder.exit();
    drop(claim_in(legacy_dir.path(), &lock).expect("the claim once the legacy holder exits"));
}

/// The open of a FIFO without `O_NONBLOCK` blocks until a writer appears.
#[test]
fn legacy_fifo_does_not_block_the_probe() {
    let dir = TempDir::new().expect("temp dir");
    make_fifo(&legacy_claim_in(&dir));
    let (tx, rx) = mpsc::channel();
    let path = dir.path().to_path_buf();
    std::thread::spawn(move || {
        let _ = tx.send(refuse_held_legacy_claim(&path));
    });
    let result = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the legacy probe must not block on a FIFO");
    result.expect("an old build made no FIFO, so the probe skips it");
}

#[test]
fn unreadable_legacy_claim_is_refused() {
    let dir = TempDir::new().expect("temp dir");
    let legacy = legacy_claim_in(&dir);
    std::fs::write(&legacy, "4242").expect("write a legacy body");
    std::fs::set_permissions(&legacy, Permissions::from_mode(0o000)).expect("chmod 0000");
    assert!(
        File::open(&legacy).is_err(),
        "this test needs a user that a 0000 file refuses"
    );
    let result = refuse_held_legacy_claim(dir.path());
    assert!(
        is_io(&result),
        "an unreadable legacy claim must refuse, got {result:?}"
    );
}

#[test]
fn legacy_symlink_is_skipped() {
    let dir = TempDir::new().expect("temp dir");
    let target = dir.path().join("target");
    std::fs::write(&target, "4242").expect("write target");
    symlink(&target, legacy_claim_in(&dir)).expect("plant a symlink");
    refuse_held_legacy_claim(dir.path()).expect("a legacy symlink is skipped");
}

#[test]
fn probe_of_a_held_claim_names_the_holder_and_writes_nothing() {
    let legacy = TempDir::new().expect("legacy dir");
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let holder = Holder::start(&lock);
    let body = std::fs::read_to_string(&lock).expect("read body");
    let result = probe_in(legacy.path(), &lock);
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("the probe must report a held claim, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    assert_eq!(std::fs::read_to_string(&lock).expect("read body"), body);
    holder.exit();
}

#[test]
fn probe_of_a_free_claim_writes_nothing() {
    let legacy = TempDir::new().expect("legacy dir");
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    probe_in(legacy.path(), &lock).expect("a missing file is a free claim");
    assert!(!lock.exists(), "the probe must not create the file");
    std::fs::write(&lock, "4242 old holder").expect("write body");
    probe_in(legacy.path(), &lock).expect("an unlocked file is a free claim");
    assert_eq!(
        std::fs::read_to_string(&lock).expect("read body"),
        "4242 old holder"
    );
}

#[test]
fn probe_refuses_while_a_legacy_claim_is_held() {
    let legacy_dir = TempDir::new().expect("legacy dir");
    let holder = Holder::start(&legacy_claim_in(&legacy_dir));
    let dir = TempDir::new().expect("temp dir");
    let result = probe_in(legacy_dir.path(), &lock_in(&dir));
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("a held legacy claim must show as held, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    holder.exit();
}

#[test]
fn legacy_directory_is_skipped() {
    let dir = TempDir::new().expect("temp dir");
    std::fs::create_dir(legacy_claim_in(&dir)).expect("plant a directory");
    refuse_held_legacy_claim(dir.path()).expect("an old build made no directory");
}

/// A hard link to a held legacy claim shares its inode, and so its flock.
#[test]
fn hard_linked_legacy_claim_is_probed() {
    let held_dir = TempDir::new().expect("held dir");
    let held = held_dir.path().join("held");
    let holder = Holder::start(&held);
    let dir = TempDir::new().expect("temp dir");
    std::fs::hard_link(&held, legacy_claim_in(&dir)).expect("hard-link the held file");
    let result = refuse_held_legacy_claim(dir.path());
    let Err(ClaimError::AlreadyHeld { holder_pid, .. }) = result else {
        panic!("a held hard-linked legacy claim must refuse, got {result:?}");
    };
    assert_eq!(holder_pid, Some(holder.pid()));
    holder.exit();
}

#[test]
fn probe_refuses_a_hard_link_at_the_claim_path() {
    let legacy = TempDir::new().expect("legacy dir");
    let dir = TempDir::new().expect("temp dir");
    let target = dir.path().join("target");
    std::fs::write(&target, "4242 some command").expect("write target");
    let lock = lock_in(&dir);
    std::fs::hard_link(&target, &lock).expect("plant a hard link");
    let result = probe_in(legacy.path(), &lock);
    let Err(ClaimError::Io { source, .. }) = result else {
        panic!("the probe must refuse a hard link, got {result:?}");
    };
    assert!(
        source.to_string().contains("more than one link"),
        "refused for another reason: {source}"
    );
}

/// A claim that is held until the process is gone outlives its guard: the
/// lock stays, as it does for a holder that still runs.
#[test]
fn a_claim_held_until_process_exit_outlives_its_guard() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let claim = claim_at(&lock).expect("first claim");
    claim.hold_until_process_exit().expect("duplicate the fd");
    drop(claim);
    assert!(
        is_held(&claim_at(&lock)),
        "the claim was released with its guard"
    );
}

/// End a runtime through `block_on_serve` as `rmlx serve` does, with `claim`
/// and one blocking job that runs until the returned sender is dropped.
/// `exit_in` is the deadline the serve future returns.
fn serve_to_the_exit(claim: &GpuClaim, exit_in: Duration) -> mpsc::Sender<()> {
    use crate::shutdown::shutdown_tests::{held_job, runtime, serving};
    use crate::shutdown::{block_on_serve_within, ExitDeadline};

    let job = held_job();
    let deadline = move || Ok(ExitDeadline(std::time::Instant::now() + exit_in));
    block_on_serve_within(
        runtime(),
        Some(claim),
        serving(job.run, job.started, deadline),
        exit_in,
    )
    .expect("the shutdown is not an error");
    assert!(
        !job.finished.load(Ordering::Acquire),
        "the job ended before the runtime did"
    );
    job.release
}

/// A shutdown that leaves a job on the blocking pool at the exit deadline
/// keeps the claim after the guard of its caller is dropped: the next process
/// cannot get it while this one still runs.
#[test]
fn a_job_left_at_the_exit_deadline_keeps_the_claim_past_its_guard() {
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let claim = GpuClaim::Metal(claim_at(&lock).expect("first claim"));

    let release = serve_to_the_exit(&claim, Duration::from_millis(100));

    drop(claim);
    let next = claim_at(&lock);
    drop(release);
    assert!(
        is_held(&next),
        "the claim was released with a job still running"
    );
}

/// A shutdown whose runtime ended before the deadline holds nothing: the
/// claim is free when the guard of its caller is dropped.
#[test]
fn a_shutdown_with_no_job_left_frees_the_claim_with_its_guard() {
    use crate::shutdown::{block_on_serve_within, ExitDeadline};

    let _no_spawn = no_spawn();
    let dir = TempDir::new().expect("temp dir");
    let lock = lock_in(&dir);
    let claim = GpuClaim::Metal(claim_at(&lock).expect("first claim"));
    let far = Duration::from_secs(120);

    block_on_serve_within(
        crate::shutdown::shutdown_tests::runtime(),
        Some(&claim),
        async move { Ok(ExitDeadline(std::time::Instant::now() + far)) },
        far,
    )
    .expect("the shutdown is not an error");

    drop(claim);
    drop(claim_at(&lock).expect("the claim is free after a clean shutdown"));
}

#[test]
fn budget_reservations_are_serialized_under_race_for_fifty_rounds() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    const THREADS: usize = 5;
    const SYNTHETIC_RAM_BYTES: u64 = 10 * 1024 * 1024 * 1024;
    const EXPECTED_LIMIT_GB: f64 = 8.5;

    for round in 0..50 {
        let dir = TempDir::new().expect("temporary ledger directory");
        let claim_path = dir.path().join("claim");
        let path = dir.path().join("budget.json");
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        #[allow(
            clippy::needless_collect,
            reason = "spawn every barrier participant before joining so the claims race concurrently"
        )]
        let workers = (0..THREADS)
            .map(|_| {
                let claim_path = claim_path.clone();
                let path = path.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    reserve_budget_coordinated_at(&claim_path, &path, 2.0, SYNTHETIC_RAM_BYTES)
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();

        let claims = workers
            .into_iter()
            .map(|worker| worker.join().expect("reservation thread did not panic"))
            .collect::<Vec<_>>();
        let admitted = claims.iter().filter(|claim| claim.is_ok()).count();
        assert_eq!(admitted, 4, "each round must admit only four 2-GiB claims");
        assert!(claims
            .iter()
            .filter_map(|claim| claim.as_ref().err())
            .all(|error| matches!(error, ClaimError::BudgetExceeded { .. })));

        let ledger: BudgetLedger =
            serde_json::from_slice(&std::fs::read(&path).expect("read ledger"))
                .expect("parse ledger");
        let allocated_gb = ledger
            .entries
            .iter()
            .map(|entry| entry.requested_gb)
            .sum::<f64>();
        assert!(
            allocated_gb <= EXPECTED_LIMIT_GB,
            "round {round}: {allocated_gb} GiB exceeds {EXPECTED_LIMIT_GB} GiB"
        );
        assert_eq!(ledger.entries.len(), 4);

        drop(claims);
        let ledger: BudgetLedger =
            serde_json::from_slice(&std::fs::read(&path).expect("read released ledger"))
                .expect("parse released ledger");
        assert!(ledger.entries.is_empty(), "all RAII claims should release");
    }
}

#[test]
fn budget_claim_drop_removes_only_its_entry() {
    let dir = TempDir::new().expect("temporary ledger directory");
    let path = dir.path().join("budget.json");
    let first = reserve_budget_at(&path, 1.0, 4 * 1024 * 1024 * 1024)
        .expect("first reservation fits under 3.4 GiB");
    let second = reserve_budget_at(&path, 1.0, 4 * 1024 * 1024 * 1024)
        .expect("second reservation fits under 3.4 GiB");
    drop(first);
    let ledger: BudgetLedger =
        serde_json::from_slice(&std::fs::read(&path).expect("read ledger")).expect("parse ledger");
    assert_eq!(ledger.entries.len(), 1);
    assert_eq!(ledger.entries[0].id, second.id);
}

#[test]
fn budget_claim_reaps_dead_pid_when_next_process_registers() {
    let dead_pid = (1_000_000..1_001_000)
        .find(|pid| matches!(rmlx_core::system::process_exists(*pid), Ok(false)))
        .expect("find a PID that does not exist");
    let dir = TempDir::new().expect("temporary ledger directory");
    let path = dir.path().join("budget.json");
    let mut ledger = BudgetLedger {
        available_gb: 3.4,
        entries: vec![BudgetEntry {
            id: "orphan".to_owned(),
            pid: dead_pid,
            requested_gb: 3.0,
            argv: vec!["rmlx".to_owned(), "serve".to_owned()],
        }],
    };
    std::fs::write(
        &path,
        serde_json::to_vec(&ledger).expect("serialize seed ledger"),
    )
    .expect("write seed ledger");
    ledger.entries.clear();

    let _claim = reserve_budget_at(&path, 1.0, 4 * 1024 * 1024 * 1024)
        .expect("orphaned reservation should be reclaimed");
    let actual: BudgetLedger =
        serde_json::from_slice(&std::fs::read(&path).expect("read ledger")).expect("parse ledger");
    assert_eq!(actual.entries.len(), 1);
    assert_ne!(actual.entries[0].id, "orphan");
}

#[test]
fn legacy_claim_path_does_not_create_a_budget_ledger() {
    let dir = TempDir::new().expect("temporary claim directory");
    let claim_path = dir.path().join("metal.claim");
    let ledger_path = dir.path().join("budget.json");
    let claim =
        claim_default_at(&claim_path, &ledger_path).expect("default exclusive claim succeeds");
    assert!(claim_path.exists());
    assert!(
        !ledger_path.exists(),
        "legacy claim must not create a ledger"
    );
    drop(claim);
}

#[test]
fn probe_refuses_while_a_budget_reservation_is_live() {
    let dir = TempDir::new().expect("temporary claim directory");
    let claim_path = dir.path().join("metal.claim");
    let ledger_path = dir.path().join("budget.json");
    let legacy_dir = dir.path().join("legacy");
    std::fs::create_dir(&legacy_dir).expect("legacy directory");
    let budget =
        reserve_budget_coordinated_at(&claim_path, &ledger_path, 1.0, 8 * 1024 * 1024 * 1024)
            .expect("budget reservation");

    let result = probe_in_with_ledger(&legacy_dir, &claim_path, Some(&ledger_path));
    assert!(
        matches!(result, Err(ClaimError::AlreadyHeld { .. })),
        "claim probe must see experimental reservations: {result:?}"
    );
    drop(budget);
    assert!(probe_in_with_ledger(&legacy_dir, &claim_path, Some(&ledger_path)).is_ok());
}

#[test]
fn default_exclusive_claim_refuses_a_live_budget_reservation() {
    let dir = TempDir::new().expect("temporary claim directory");
    let claim_path = dir.path().join("rmlx.claim");
    let ledger_path = dir.path().join("rmlx.budget.json");
    let budget =
        reserve_budget_coordinated_at(&claim_path, &ledger_path, 1.0, 8 * 1024 * 1024 * 1024)
            .expect("budget claim is admitted");

    let refused = claim_default_at(&claim_path, &ledger_path);
    let Err(ClaimError::AlreadyHeld {
        holder_pid,
        holder_command,
        ..
    }) = refused
    else {
        panic!("default claim must refuse a live budget holder, got {refused:?}");
    };
    assert_eq!(holder_pid, Some(std::process::id()));
    assert!(holder_command.contains("rmlx"));

    drop(budget);
    drop(claim_default_at(&claim_path, &ledger_path).expect("default claim after budget drop"));
}

#[test]
fn budget_reservation_refuses_a_live_default_exclusive_claim() {
    let dir = TempDir::new().expect("temporary claim directory");
    let claim_path = dir.path().join("rmlx.claim");
    let ledger_path = dir.path().join("rmlx.budget.json");
    let exclusive = claim_at(&claim_path).expect("default claim acquired");

    let refused =
        reserve_budget_coordinated_at(&claim_path, &ledger_path, 1.0, 8 * 1024 * 1024 * 1024);
    assert!(
        matches!(refused, Err(ClaimError::AlreadyHeld { .. })),
        "budget path must refuse while default claim is held: {refused:?}"
    );
    assert!(
        !ledger_path.exists(),
        "a refused budget claim must not create the ledger"
    );
    drop(exclusive);
}

#[test]
fn default_and_budget_claim_race_has_one_winner_for_fifty_rounds() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    enum Winner {
        Default(MetalClaim),
        Budget(MemoryBudgetClaim),
    }

    for round in 0..10 {
        let dir = TempDir::new().expect("temporary claim directory");
        let claim_path = dir.path().join("rmlx.claim");
        let ledger_path = dir.path().join("rmlx.budget.json");
        let barrier = Arc::new(Barrier::new(3));
        let default_path = claim_path.clone();
        let default_ledger = ledger_path.clone();
        let default_barrier = Arc::clone(&barrier);
        let default = thread::spawn(move || {
            default_barrier.wait();
            claim_default_at(&default_path, &default_ledger).map(Winner::Default)
        });
        let budget_barrier = Arc::clone(&barrier);
        let budget = thread::spawn(move || {
            budget_barrier.wait();
            reserve_budget_coordinated_at(&claim_path, &ledger_path, 1.0, 8 * 1024 * 1024 * 1024)
                .map(Winner::Budget)
        });
        barrier.wait();

        let default_result = default.join().expect("default thread did not panic");
        let budget_result = budget.join().expect("budget thread did not panic");
        let winners = usize::from(default_result.is_ok()) + usize::from(budget_result.is_ok());
        assert_eq!(winners, 1, "round {round}: mixed-mode arbitration failed");
        assert!(
            matches!(default_result.as_ref(), Err(ClaimError::AlreadyHeld { .. }))
                || matches!(budget_result.as_ref(), Err(ClaimError::AlreadyHeld { .. })),
            "losing mode must fail closed"
        );
        match (default_result, budget_result) {
            (Ok(Winner::Default(claim)), Err(_)) => drop(claim),
            (Err(_), Ok(Winner::Budget(claim))) => drop(claim),
            _ => panic!("exactly one valid winner is required"),
        }
    }
}

#[test]
fn invalid_ledger_entry_fails_closed_instead_of_reducing_allocated_total() {
    let dir = TempDir::new().expect("temporary ledger directory");
    let path = dir.path().join("budget.json");
    let forged = BudgetLedger {
        available_gb: 8.5,
        entries: vec![BudgetEntry {
            id: "forged-negative".to_owned(),
            pid: std::process::id(),
            requested_gb: -100.0,
            argv: Vec::new(),
        }],
    };
    std::fs::write(
        &path,
        serde_json::to_vec(&forged).expect("serialize ledger"),
    )
    .expect("write ledger");

    assert!(
        matches!(
            reserve_budget_at(&path, 1.0, 10 * 1024 * 1024 * 1024),
            Err(ClaimError::Io { source, .. }) if source.kind() == io::ErrorKind::InvalidData
        ),
        "invalid ledger entries must fail closed"
    );
}

#[test]
fn atomic_ledger_write_failure_preserves_the_previous_ledger() {
    let dir = TempDir::new().expect("temporary ledger directory");
    let path = dir.path().join("budget.json");
    let old = BudgetLedger {
        available_gb: 8.5,
        entries: vec![BudgetEntry {
            id: "live-before-failure".to_owned(),
            pid: std::process::id(),
            requested_gb: 2.0,
            argv: vec!["rmlx".to_owned(), "serve".to_owned()],
        }],
    };
    let old_bytes = serde_json::to_vec_pretty(&old).expect("serialize old ledger");
    std::fs::write(&path, &old_bytes).expect("write old ledger");
    let new = BudgetLedger {
        available_gb: 8.5,
        entries: Vec::new(),
    };
    let (file, created) = open_budget_file(&path).expect("open ledger");
    assert!(!created);
    file.lock().expect("lock ledger");

    let result = write_budget_ledger_with_hook(&path, &new, || {
        Err(io::Error::other("injected failure before publish"))
    });
    assert!(matches!(result, Err(ClaimError::Io { .. })));
    assert_eq!(
        std::fs::read(&path).expect("read retained ledger"),
        old_bytes,
        "failed transaction must leave the live ledger byte-for-byte intact"
    );
    assert_eq!(
        std::fs::read_dir(dir.path())
            .expect("list ledger directory")
            .count(),
        1,
        "temporary file should be removed after an ordinary write failure"
    );
}

#[test]
fn existing_empty_ledger_fails_closed() {
    let dir = TempDir::new().expect("temporary ledger directory");
    let path = dir.path().join("budget.json");
    std::fs::write(&path, []).expect("create empty ledger");
    assert!(matches!(
        reserve_budget_at(&path, 1.0, 8 * 1024 * 1024 * 1024),
        Err(ClaimError::Io { source, .. }) if source.kind() == io::ErrorKind::InvalidData
    ));
}

struct BudgetHolder {
    child: Child,
    stdin: Option<ChildStdin>,
    pid: u32,
}

impl BudgetHolder {
    fn start(claim_path: &Path, ledger_path: &Path, requested_gb: f64) -> Self {
        let spawn = no_spawn();
        let mut child = Command::new(std::env::current_exe().expect("current test binary"))
            .args([
                "claim::tests::budget_claim_holder_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start budget holder child");
        drop(spawn);
        let pid = child.id();
        let mut stdin = child.stdin.take().expect("child stdin");
        writeln!(stdin, "{}", claim_path.display()).expect("send claim path");
        writeln!(stdin, "{}", ledger_path.display()).expect("send ledger path");
        writeln!(stdin, "{requested_gb}").expect("send reservation");
        stdin.flush().expect("flush holder configuration");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let registered = std::fs::read(ledger_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<BudgetLedger>(&bytes).ok())
                .is_some_and(|ledger| ledger.entries.iter().any(|entry| entry.pid == pid));
            if registered {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "child {pid} did not register its reservation"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        BudgetHolder {
            child,
            stdin: Some(stdin),
            pid,
        }
    }

    fn kill(mut self) {
        self.child.kill().expect("kill budget holder");
        self.child.wait().expect("wait for killed budget holder");
    }

    fn release(mut self) {
        drop(self.stdin.take());
        let status = self.child.wait().expect("wait for released holder");
        assert!(status.success(), "budget holder failed: {status}");
    }
}

#[test]
#[ignore = "child process of the multi-process budget reservation test"]
fn budget_claim_holder_child() {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        return;
    }
    let mut input = stdin.lock();
    let mut claim_path = String::new();
    let mut ledger_path = String::new();
    let mut requested_gb = String::new();
    if input.read_line(&mut claim_path).is_err()
        || input.read_line(&mut ledger_path).is_err()
        || input.read_line(&mut requested_gb).is_err()
    {
        return;
    }
    let claim_path = PathBuf::from(claim_path.trim());
    let ledger_path = PathBuf::from(ledger_path.trim());
    let Ok(requested_gb) = requested_gb.trim().parse::<f64>() else {
        return;
    };
    let Ok(claim) = reserve_budget_coordinated_at(
        &claim_path,
        &ledger_path,
        requested_gb,
        10 * 1024 * 1024 * 1024,
    ) else {
        return;
    };
    println!("held {}", std::process::id());
    io::stdout().flush().expect("flush held status");
    let mut rest = Vec::new();
    let _ = input.read_to_end(&mut rest);
    drop(claim);
}

#[test]
fn separate_process_reservations_are_reaped_after_sigkill() {
    let dir = TempDir::new().expect("temporary coordination directory");
    let claim_path = dir.path().join("rmlx.claim");
    let ledger_path = dir.path().join("rmlx.budget.json");
    let first = BudgetHolder::start(&claim_path, &ledger_path, 2.0);
    let second = BudgetHolder::start(&claim_path, &ledger_path, 2.0);
    assert_ne!(first.pid, second.pid);

    let ledger: BudgetLedger =
        serde_json::from_slice(&std::fs::read(&ledger_path).expect("read live ledger"))
            .expect("parse live ledger");
    let pids = ledger
        .entries
        .iter()
        .map(|entry| entry.pid)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(pids.len(), 2, "ledger must hold entries from two PIDs");
    assert!(pids.contains(&first.pid) && pids.contains(&second.pid));

    let dead_pid = first.pid;
    first.kill();
    let own =
        reserve_budget_coordinated_at(&claim_path, &ledger_path, 2.0, 10 * 1024 * 1024 * 1024)
            .expect("registration reaps the SIGKILLed process before admission");
    let ledger: BudgetLedger =
        serde_json::from_slice(&std::fs::read(&ledger_path).expect("read reaped ledger"))
            .expect("parse reaped ledger");
    assert_eq!(ledger.entries.len(), 2);
    assert!(ledger.entries.iter().all(|entry| entry.pid != dead_pid));
    assert!(ledger.entries.iter().any(|entry| entry.pid == second.pid));
    assert!(ledger
        .entries
        .iter()
        .any(|entry| entry.pid == std::process::id()));

    drop(own);
    second.release();
    let ledger: BudgetLedger =
        serde_json::from_slice(&std::fs::read(&ledger_path).expect("read cleared ledger"))
            .expect("parse cleared ledger");
    assert!(ledger.entries.is_empty());
}
