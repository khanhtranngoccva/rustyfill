use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::AtomicBool;

use super::RandomError;

unsafe extern "C" {
    fn randSecure() -> libc::c_int;
    fn usleep(microseconds: libc::c_ulong);
    fn randABytes(buf: *mut u8, len: libc::c_int) -> libc::c_int;
}

static RNG_INIT: AtomicBool = AtomicBool::new(false);

pub fn fill_bytes(mut bytes: &mut [u8]) -> Result<(), RandomError> {
    while !RNG_INIT.load(Relaxed) {
        let ret = unsafe { randSecure() };
        if ret < 0 {
            return Err(RandomError::Platform(
                "VxWorks randSecure failed".into(),
            ));
        } else if ret > 0 {
            RNG_INIT.store(true, Relaxed);
            break;
        }

        unsafe { usleep(10) };
    }

    while !bytes.is_empty() {
        let len = bytes.len().try_into().unwrap_or(libc::c_int::MAX);
        let ret = unsafe { randABytes(bytes.as_mut_ptr(), len) };
        if ret < 0 {
            return Err(RandomError::Platform(
                "VxWorks randABytes failed".into(),
            ));
        }
        bytes = &mut bytes[len as usize..];
    }
    Ok(())
}
