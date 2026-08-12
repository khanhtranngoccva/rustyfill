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

use lang_std::borrow::Cow;
use lang_std::fs::File;
use lang_std::io::{self, Read};
use lang_std::os::fd::AsRawFd;
use lang_std::sync::atomic::AtomicBool;
use lang_std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use lang_core::mem;
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

fn getrandom_impl(mut bytes: &mut [u8], insecure: bool) -> Result<(), RandomError> {
    static GRND_INSECURE_AVAILABLE: AtomicBool = AtomicBool::new(true);
    static URANDOM_READY: AtomicBool = AtomicBool::new(false);
    static DEVICE: OnceCell<File> = OnceCell::new();

    // Try the dynamically-resolved `getrandom` symbol first.
    // (`getrandom` was added in glibc 2.25, musl 1.1.20, android API level 28)
    if let Some(getrandom_fn) = resolve_getrandom() {
        loop {
            if bytes.is_empty() {
                return Ok(());
            }

            let flags = if insecure {
                if GRND_INSECURE_AVAILABLE.load(Relaxed) {
                    libc::GRND_INSECURE
                } else {
                    libc::GRND_NONBLOCK
                }
            } else {
                0
            };

            let ret = unsafe {
                getrandom_fn(bytes.as_mut_ptr().cast(), bytes.len(), flags)
            };
            if ret != -1 {
                bytes = &mut bytes[ret as usize..];
            } else {
                let err = io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or(libc::EIO);
                match err {
                    libc::EINTR => continue,
                    // `GRND_INSECURE` is not available, try
                    // `GRND_NONBLOCK`.
                    libc::EINVAL if flags == libc::GRND_INSECURE => {
                        GRND_INSECURE_AVAILABLE.store(false, Relaxed);
                        continue;
                    }
                    // The pool is not initialized yet, fall back to
                    // /dev/urandom for now.
                    libc::EAGAIN if flags == libc::GRND_NONBLOCK => break,
                    // `getrandom` is unavailable or blocked by seccomp.
                    // Don't try it again and fall back to /dev/urandom.
                    libc::ENOSYS | libc::EPERM => break,
                    other => {
                        return Err(RandomError::Syscall(other));
                    }
                }
            }
        }
    }

    // When we want cryptographic strength, we need to wait for the CPRNG-pool
    // to become initialized. Do this by polling `/dev/random` until it is ready.
    if !insecure && !URANDOM_READY.load(Acquire) {
        let random = File::open("/dev/random").map_err(|_| {
            RandomError::Platform(Cow::Borrowed("failed to open /dev/random"))
        })?;
        let mut fd = libc::pollfd {
            fd: random.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };

        while !URANDOM_READY.load(Acquire) {
            let ret = unsafe { libc::poll(&mut fd, 1, -1) };
            match ret {
                1 => {
                    assert_eq!(fd.revents, libc::POLLIN);
                    URANDOM_READY.store(true, Release);
                    break;
                }
                -1 if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) => {
                    continue;
                }
                _ => {
                    return Err(RandomError::Platform(Cow::Borrowed(
                        "poll(\"/dev/random\") failed",
                    )))
                }
            }
        }
    }

    let dev = DEVICE
        .get_or_try_init(|| File::open("/dev/urandom"))
        .map_err(|_| {
            RandomError::Platform(Cow::Borrowed("failed to open /dev/urandom"))
        })?;
    let mut dev = dev;
    dev.read_exact(bytes).map_err(|_| {
        RandomError::Platform(Cow::Borrowed("failed to read from /dev/urandom"))
    })?;
    Ok(())
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
