use super::RandomError;

unsafe extern "C" {
    fn SOLID_RNG_SampleRandomBytes(buffer: *mut u8, length: usize) -> c_int;
}

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe {
        let result = SOLID_RNG_SampleRandomBytes(bytes.as_mut_ptr(), bytes.len());
        if result == 0 {
            Ok(())
        } else {
            Err(RandomError::Platform(core::borrow::Cow::Borrowed(
                "SOLID_RNG_SampleRandomBytes failed",
            )))
        }
    }
}
