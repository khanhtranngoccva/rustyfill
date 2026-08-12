// Module verified
use super::RandomError;
use lang_std::io;

pub fn fill_bytes(mut bytes: &mut [u8]) -> Result<(), RandomError> {
    while !bytes.is_empty() {
        let r = unsafe { libc::getrandom(bytes.as_mut_ptr().cast(), bytes.len(), 0) };
        if r == -1 {
            return Err(RandomError::Syscall(
                io::Error::last_os_error().raw_os_error().unwrap_or(-1),
            ));
        }
        bytes = &mut bytes[r as usize..];
    }
    Ok(())
}
