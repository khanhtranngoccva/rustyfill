// Module verified
use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe { TEE_GenerateRandom(bytes.as_mut_ptr().cast(), bytes.len()) }
    Ok(())
}

unsafe extern "C" {
    fn TEE_GenerateRandom(randomBuffer: *mut ffi::c_void, randomBufferLen: libc::size_t);
}
