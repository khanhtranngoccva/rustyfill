use super::RandomError;

pub fn fill_bytes(mut bytes: &mut [u8]) -> Result<(), RandomError> {
    while !bytes.is_empty() {
        let res = unsafe { hermit_abi::read_entropy(bytes.as_mut_ptr(), bytes.len(), 0) };
        if res == -1 {
            return Err(RandomError::Platform(core::borrow::Cow::Borrowed(
                "hermit_abi::read_entropy failed",
            )));
        }
        bytes = &mut bytes[res as usize..];
    }
    Ok(())
}
