use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe {
        let ret = wasip1::random_get(bytes.as_mut_ptr(), bytes.len());
        if ret == 0 {
            Ok(())
        } else {
            Err(RandomError::Syscall(ret))
        }
    }
}
