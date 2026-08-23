//! Random data generation with the Linux kernel.
//!
//! The first interface random data interface to be introduced on Linux were
//! the `/dev/random` and `/dev/urandom` special files. As paths can become
//! unreachable when inside a chroot and when the file descriptors are exhausted,
//! this was not enough to provide userspace with a reliable source of randomness,
//! so when the OpenBSD 5.6 introduced the `getentropy` syscall, Linux 3.17 got
//! its very own `getrandom`  syscall to match.[^1] Unfortunately, even if our
//! minimum supported version were high enough, we still couldn't rely on the
//! syscall being available, as it is blocked in `seccomp` by default.
//!
//! The question is therefore which of the random sources to use. Historically,
//! the kernel contained two pools: the blocking and non-blocking pool. The
//! blocking pool used entropy estimation to limit the amount of available
//! bytes, while the non-blocking pool, once initialized using the blocking
//! pool, uses a CPRNG to return an unlimited number of random bytes. With a
//! strong enough CPRNG however, the entropy estimation didn't contribute that
//! much towards security while being an excellent vector for DoS attacks. Thus,
//! the blocking pool was removed in kernel version 5.6.[^2] That patch did not
//! magically increase the quality of the non-blocking pool, however, so we can
//! safely consider it strong enough even in older kernel versions and use it
//! unconditionally.
//!
//! One additional consideration to make is that the non-blocking pool is not
//! always initialized during early boot. We want the best quality of randomness
//! for the output of `SystemRng` so we simply wait until it is initialized.
//! When `HashMap` keys however, this represents a potential source of
//! deadlocks, as the additional entropy may only be generated once the program
//! makes forward progress. In that case, we just use the best random data the
//! system has available at the time.
//!
//! So in conclusion, we always want the output of the non-blocking pool, but
//! may need to wait until it is initialized. The default behavior of `getrandom`
//! is to wait until the non-blocking pool is initialized and then draw from there,
//! so if `getrandom` is available, we use its default to generate the bytes. For
//! `HashMap`, however, we need to specify the `GRND_INSECURE` flags, but that
//! is only available starting with kernel version 5.6. Thus, if we detect that
//! the flag is unsupported, we try `GRND_NONBLOCK` instead, which will only
//! succeed if the pool is initialized. If it isn't, we fall back to the file
//! access method.
//!
//! The behavior of `/dev/urandom` is inverse to that of `getrandom`: it always
//! yields data, even when the pool is not initialized. For generating `HashMap`
//! keys, this is not important, so we can use it directly. For secure data
//! however, we need to wait until initialization, which we can do by `poll`ing
//! `/dev/random`.
//!
//! TLDR: our fallback strategies are:
//!
//! Secure data                                 | `HashMap` keys
//! --------------------------------------------|------------------
//! getrandom(0)                                | getrandom(GRND_INSECURE)
//! poll("/dev/random") && read("/dev/urandom") | getrandom(GRND_NONBLOCK)
//!                                             | read("/dev/urandom")
//!
//! [^1]: <https://lwn.net/Articles/606141/>
//! [^2]: <https://lwn.net/Articles/808575/>
//!
// TODO: once the minimum supported kernel version is 5.6+, remove the
// `GRND_NONBLOCK` fallback and use `/dev/random` instead of `/dev/urandom`
// when secure data is required.

use lang_core::mem;
use lang_std::borrow::Cow;
use lang_std::fs::File;
use lang_std::io::{self, Read};
use lang_std::os::fd::AsRawFd;
use lang_std::sync::atomic::AtomicBool;
use lang_std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
// Fallback for get_or_try_init API
use once_cell::sync::OnceCell;

use super::RandomError;

/// Type alias for the `getrandom` C library function.
type GetrandomFn = unsafe extern "C-unwind" fn(*mut u8, usize, u32) -> isize;

/// Dynamically resolve the `getrandom` symbol from the process's linked libc.
///
/// Uses `dlsym(RTLD_DEFAULT, ...)` so no explicit path or `dlopen` is needed —
/// it searches the global symbol table of all loaded objects (main executable +
/// shared libraries). Returns `None` if the symbol is not found (e.g., glibc < 2.25,
/// musl < 1.1.20).
static GETRANDOM_FN: OnceCell<Option<GetrandomFn>> = OnceCell::new();

fn resolve_getrandom() -> Option<GetrandomFn> {
    let result = GETRANDOM_FN.get_or_init(|| {
        // SAFETY: RTLD_DEFAULT searches all currently loaded objects.
        // dlsym never frees the returned pointer; it remains valid for the
        // lifetime of the process. The cast to our function type is safe
        // because the signature matches the C declaration of `getrandom`.
        let sym = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"getrandom".as_ptr().cast()) };
        if sym.is_null() {
            None
        } else {
            Some(unsafe { mem::transmute::<*mut _, GetrandomFn>(sym) })
        }
    });
    // Function pointers are Copy, so we can safely copy out of the reference.
    *result
}

/// Fills `bytes` using the dynamically-resolved `getrandom` syscall.
///
/// Returns `Ok(true)` if the buffer was fully filled (or was empty), `Ok(false)`
/// to fall through to the `/dev/urandom` path, and `Err` on a fatal syscall
/// failure. Handles the retry/fallback dance for `EINTR`, unsupported
/// `GRND_INSECURE`, uninitialized pool (`EAGAIN`), and seccomp-blocked calls.
fn getrandom_syscall(
    bytes: &mut [u8],
    insecure: bool,
    grnd_insecure_available: &AtomicBool,
) -> Result<bool, RandomError> {
    let Some(getrandom_fn) = resolve_getrandom() else {
        return Ok(false);
    };

    let mut rest = bytes;
    loop {
        if rest.is_empty() {
            return Ok(true);
        }

        let flags = if insecure {
            if grnd_insecure_available.load(Relaxed) {
                libc::GRND_INSECURE
            } else {
                libc::GRND_NONBLOCK
            }
        } else {
            0
        };

        let ret = unsafe { getrandom_fn(rest.as_mut_ptr().cast(), rest.len(), flags) };
        if ret != -1 {
            rest = &mut rest[ret as usize..];
        } else {
            let err = io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO);
            match err {
                libc::EINTR => continue,
                // `GRND_INSECURE` is not available, try `GRND_NONBLOCK`.
                libc::EINVAL if flags == libc::GRND_INSECURE => {
                    grnd_insecure_available.store(false, Relaxed);
                    continue;
                }
                // The pool is not initialized yet, fall back to /dev/urandom.
                libc::EAGAIN if flags == libc::GRND_NONBLOCK => return Ok(false),
                // `getrandom` is unavailable or blocked by seccomp.
                libc::ENOSYS | libc::EPERM => return Ok(false),
                other => return Err(RandomError::Syscall(other)),
            }
        }
    }
}

/// Waits until the kernel's non-blocking random pool is initialized by polling
/// `/dev/random`. No-op when `already_ready` is set or when `insecure` data is
/// being requested (the pool need not be ready for hashmap seeds).
fn wait_for_urandom_ready(insecure: bool, urandom_ready: &AtomicBool) -> Result<(), RandomError> {
    if insecure || urandom_ready.load(Acquire) {
        return Ok(());
    }

    let random = File::open("/dev/random")
        .map_err(|_| RandomError::Platform(Cow::Borrowed("failed to open /dev/random")))?;
    let mut fd = libc::pollfd {
        fd: random.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };

    while !urandom_ready.load(Acquire) {
        let ret = unsafe { libc::poll(&mut fd, 1, -1) };
        match ret {
            1 => {
                assert_eq!(fd.revents, libc::POLLIN);
                urandom_ready.store(true, Release);
                break;
            }
            -1 if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) => continue,
            _ => {
                return Err(RandomError::Platform(Cow::Borrowed(
                    "poll(\"/dev/random\") failed",
                )));
            }
        }
    }
    Ok(())
}

/// Reads exactly `bytes.len()` bytes from the cached `/dev/urandom` handle.
fn read_urandom(bytes: &mut [u8], device: &OnceCell<File>) -> Result<(), RandomError> {
    let dev = device
        .get_or_try_init(|| File::open("/dev/urandom"))
        .map_err(|_| RandomError::Platform(Cow::Borrowed("failed to open /dev/urandom")))?;
    let mut dev = dev;
    dev.read_exact(bytes)
        .map_err(|_| RandomError::Platform(Cow::Borrowed("failed to read from /dev/urandom")))
}

fn getrandom_impl(bytes: &mut [u8], insecure: bool) -> Result<(), RandomError> {
    static GRND_INSECURE_AVAILABLE: AtomicBool = AtomicBool::new(true);
    static URANDOM_READY: AtomicBool = AtomicBool::new(false);
    static DEVICE: OnceCell<File> = OnceCell::new();

    // Try the dynamically-resolved `getrandom` symbol first.
    // (`getrandom` was added in glibc 2.25, musl 1.1.20, android API level 28)
    if getrandom_syscall(bytes, insecure, &GRND_INSECURE_AVAILABLE)? {
        return Ok(());
    }

    // When we want cryptographic strength, we need to wait for the CPRNG-pool
    // to become initialized before drawing from /dev/urandom.
    wait_for_urandom_ready(insecure, &URANDOM_READY)?;
    read_urandom(bytes, &DEVICE)
}

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    getrandom_impl(bytes, false)
}

pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    let mut bytes = [0; 16];
    getrandom_impl(&mut bytes, true)?;
    let k1 = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
    let k2 = u64::from_ne_bytes(bytes[8..].try_into().unwrap());
    Ok((k1, k2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lang_std::sync::atomic::AtomicBool as TestAtomicBool;

    // ── Public API: fill_bytes / hashmap_random_keys ────────────────────────

    #[test]
    fn fill_bytes_small_buffer_succeeds() {
        let mut buf = [0u8; 16];
        fill_bytes(&mut buf).expect("getrandom should work on Linux");
    }

    #[test]
    fn fill_bytes_large_buffer_succeeds() {
        let mut buf = lang_alloc::vec![0u8; 4096];
        fill_bytes(&mut buf).expect("large buffer fill should succeed");
        // Sanity: not all zeros (astronomically unlikely to be random-zero).
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn fill_bytes_empty_buffer_is_noop() {
        let mut buf: [u8; 0] = [];
        fill_bytes(&mut buf).expect("empty buffer should be a no-op success");
    }

    #[test]
    fn fill_bytes_produces_varying_output() {
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        fill_bytes(&mut a).unwrap();
        fill_bytes(&mut b).unwrap();
        // Two independent draws should differ (collision probability ~ 2^-512).
        assert_ne!(a, b);
    }

    #[test]
    fn hashmap_random_keys_returns_two_u64s() {
        let (k1, k2) = hashmap_random_keys().expect("hashmap keys should succeed");
        // Not required to differ, but both should be produced without error.
        let _ = (k1, k2);
    }

    #[test]
    fn hashmap_random_keys_varies_across_calls() {
        let (a1, a2) = hashmap_random_keys().unwrap();
        let (b1, b2) = hashmap_random_keys().unwrap();
        assert_ne!((a1, a2), (b1, b2));
    }

    // ── getrandom_syscall helper ─────────────────────────────────────────────

    #[test]
    fn syscall_fills_secure_bytes() {
        let flag = TestAtomicBool::new(true);
        let mut buf = [0u8; 8];
        let filled = getrandom_syscall(&mut buf, false, &flag).expect("syscall should succeed");
        assert!(filled, "secure path should fill via syscall");
    }

    #[test]
    fn syscall_fills_insecure_bytes() {
        let flag = TestAtomicBool::new(true);
        let mut buf = [0u8; 8];
        let filled =
            getrandom_syscall(&mut buf, true, &flag).expect("insecure syscall should succeed");
        assert!(filled);
    }

    #[test]
    fn syscall_empty_buffer_reports_filled() {
        let flag = TestAtomicBool::new(true);
        let mut buf: [u8; 0] = [];
        let filled = getrandom_syscall(&mut buf, false, &flag).unwrap();
        assert!(filled, "empty buffer is trivially 'fully filled'");
    }

    #[test]
    fn syscall_partial_then_complete_fill() {
        // Request enough that the kernel may return in chunks; the loop must
        // keep going until the whole buffer is filled.
        let flag = TestAtomicBool::new(true);
        let mut buf = lang_alloc::vec![0u8; 1 << 20]; // 1 MiB
        let filled = getrandom_syscall(&mut buf, false, &flag).unwrap();
        assert!(filled);
        assert!(buf.iter().any(|&b| b != 0));
    }

    // ── wait_for_urandom_ready helper ────────────────────────────────────────

    #[test]
    fn urandom_ready_short_circuits_when_insecure() {
        // Insecure mode never polls /dev/random, so it must succeed immediately
        // even if the ready flag is unset.
        let ready = TestAtomicBool::new(false);
        wait_for_urandom_ready(true, &ready).expect("insecure should skip polling");
        // Flag remains untouched in insecure mode.
        assert!(!ready.load(Relaxed));
    }

    #[test]
    fn urandom_ready_short_circuits_when_already_ready() {
        let ready = TestAtomicBool::new(true);
        wait_for_urandom_ready(false, &ready).expect("already-ready should be a no-op");
    }

    #[test]
    fn urandom_ready_blocks_until_pool_initialized() {
        // Secure mode with the flag unset will poll /dev/random until the
        // kernel signals readiness, then set the flag. On a normal dev box the
        // pool is already initialized, so this returns promptly.
        let ready = TestAtomicBool::new(false);
        wait_for_urandom_ready(false, &ready).expect("should wait and succeed");
        assert!(
            ready.load(Acquire),
            "ready flag must be set after successful poll"
        );
    }

    // ── read_urandom helper ──────────────────────────────────────────────────

    #[test]
    fn read_urandom_fills_exact_length() {
        static DEV: OnceCell<File> = OnceCell::new();
        let mut buf = [0u8; 64];
        read_urandom(&mut buf, &DEV).expect("urandom read should succeed");
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn read_urandom_reuses_cached_handle() {
        // Second call reuses the cached File; must still succeed.
        static DEV: OnceCell<File> = OnceCell::new();
        let mut a = [0u8; 8];
        let mut b = [0u8; 8];
        read_urandom(&mut a, &DEV).unwrap();
        read_urandom(&mut b, &DEV).unwrap();
    }
}
