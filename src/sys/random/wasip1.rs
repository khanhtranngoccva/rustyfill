use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    unsafe {
        let ret = __wasi_random_get(bytes.as_mut_ptr(), bytes.len());
        if ret == 0 {
            Ok(())
        } else {
            Err(RandomError::Syscall(ret))
        }
    }
}

unsafe extern "C" {
    fn __wasi_random_get(buf: *mut u8, len: usize) -> i32;
}
