// Module verified
use super::RandomError;

unsafe extern "C" {
    fn esp_fill_random(buf: *mut ffi::c_void, len: usize);
}

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe { esp_fill_random(bytes.as_mut_ptr().cast(), bytes.len()) }
    Ok(())
}
