// Module verified
use lang_std::sync::atomic::AtomicBool;
use lang_std::sync::atomic::Ordering::Relaxed;

use super::RandomError;

static RNG_INIT: AtomicBool = AtomicBool::new(false);

pub fn fill_bytes(mut bytes: &mut [u8]) -> Result<(), RandomError> {
    while !RNG_INIT.load(Relaxed) {
        let ret = unsafe { libc::randSecure() };
        if ret < 0 {
            return Err(RandomError::Platform(
                 lang_alloc::borrow::Cow::Borrowed("VxWorks randSecure failed"),
            ));
        } else if ret > 0 {
            RNG_INIT.store(true, Relaxed);
            break;
        }

        unsafe { libc::usleep(10) };
    }

    while !bytes.is_empty() {
        let len = bytes.len().try_into().unwrap_or(libc::c_int::MAX);
        let ret = unsafe { libc::randABytes(bytes.as_mut_ptr(), len) };
        if ret < 0 {
            return Err(RandomError::Platform(
                 lang_alloc::borrow::Cow::Borrowed("VxWorks randABytes failed"),
            ));
        }
        bytes = &mut bytes[len as usize..];
    }
    Ok(())
}
