//! Small operating-system probes used by process coordination.

use std::io;

/// Return the machine's physical RAM in bytes.
///
/// # Errors
/// Returns an OS error when the platform does not expose the physical-memory
/// value or the query fails.
pub fn physical_memory_bytes() -> io::Result<u64> {
    physical_memory::read()
}

/// Return whether a PID currently exists.
///
/// Permission denial means the process exists but is not visible to the
/// caller. Other OS errors are returned rather than treated as a dead PID.
///
/// # Errors
/// Returns an OS error when process existence cannot be determined.
pub fn process_exists(pid: u32) -> io::Result<bool> {
    process::exists(pid)
}

#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    reason = "sysctlbyname is the macOS API for physical RAM and has no safe std equivalent"
)]
mod physical_memory {
    use std::io;
    use std::mem::size_of;

    pub(super) fn read() -> io::Result<u64> {
        let mut bytes = 0_u64;
        let mut size = size_of::<u64>();
        // SAFETY: `hw.memsize` is a valid NUL-terminated sysctl name; `bytes`
        // is writable for `size_of::<u64>()` bytes and `size` is writable.
        let result = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        if size != size_of::<u64>() || bytes == 0 {
            return Err(io::Error::other("hw.memsize returned an invalid value"));
        }
        Ok(bytes)
    }
}

#[cfg(not(target_os = "macos"))]
mod physical_memory {
    use std::io;

    pub(super) fn read() -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "physical RAM query is only implemented on macOS",
        ))
    }
}

#[cfg(unix)]
#[allow(
    unsafe_code,
    reason = "libc::kill(pid, 0) is the required Unix process-liveness probe"
)]
mod process {
    use std::io;

    pub(super) fn exists(pid: u32) -> io::Result<bool> {
        let pid = libc::pid_t::try_from(pid)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "PID is out of range"))?;
        // SAFETY: signal 0 performs no signal delivery; `pid` is a checked
        // positive process identifier.
        let result = unsafe { libc::kill(pid, 0) };
        if result == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(false),
            Some(libc::EPERM) => Ok(true),
            _ => Err(error),
        }
    }
}

#[cfg(not(unix))]
mod process {
    use std::io;

    pub(super) fn exists(_pid: u32) -> io::Result<bool> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process probing is only implemented on Unix",
        ))
    }
}
