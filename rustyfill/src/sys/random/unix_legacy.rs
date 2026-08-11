//! Random data from `/dev/urandom`
//!
//! Before `getentropy` was standardized in 2024, UNIX didn't have a standardized
//! way of getting random data, so systems just followed the precedent set by
//! Linux and exposed random devices at `/dev/random` and `/dev/urandom`. Thus,
//! for the few systems that support neither `arc4random_buf` nor `getentropy`
//! yet, we just read from the file.

// Module verified
use super::RandomError;
use crate::lang_std::fs::File;
use crate::lang_std::io::Read;
// Fallback for get_or_try_init API
use once_cell::sync::OnceCell;

static DEVICE: OnceCell<File> = OnceLock::new();

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    let dev = DEVICE
        .get_or_try_init(|| File::open("/dev/urandom"))
        .map_err(|_| RandomError::Platform (borrow::Cow::Borrowed("failed to open /dev/urandom")))?;
    let mut dev = dev;
    dev.read_exact(bytes)
        .map_err(|_| RandomError::Platform (borrow::Cow::Borrowed("failed to read from /dev/urandom")))?;
    Ok(())
}
