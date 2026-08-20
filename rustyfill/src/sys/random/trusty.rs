// Module verified
use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe { trusty_rng_secure_rand(bytes.as_mut_ptr().cast(), bytes.len()) }
    Ok(())
}

unsafe extern "C" {
    fn trusty_rng_secure_rand(randomBuffer: *mut ffi::c_void, randomBufferLen: libc::size_t);
}
