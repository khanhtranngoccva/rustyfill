use super::RandomError;

#[cfg(not(target_vendor = "win7"))]
#[inline]
pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    let ret = unsafe {
        ProcessPrng(
            bytes.as_mut_ptr(),
            bytes.len().try_into().unwrap_or(u32::MAX),
        )
    };
    // ProcessPrng is documented as always returning `TRUE`.
    // https://learn.microsoft.com/en-us/windows/win32/seccng/processprng#return-value
    debug_assert_ne!(ret, 0);
    Ok(())
}

#[cfg(target_vendor = "win7")]
pub fn fill_bytes(mut bytes: &mut [u8]) -> Result<(), RandomError> {
    while !bytes.is_empty() {
        let len = bytes.len().try_into().unwrap_or(u32::MAX);
        let ret = unsafe { RtlGenRandom(bytes.as_mut_ptr().cast(), len) };
        if ret == FALSE {
            return Err(RandomError::Platform("RtlGenRandom failed".into()));
        }
        bytes = &mut bytes[len as usize..];
    }
    Ok(())
}

#[cfg(not(target_vendor = "win7"))]
unsafe extern "system" {
    fn ProcessPrng(random_buffer: *mut u8, length: u32) -> BOOLEAN;
}

#[cfg(target_vendor = "win7")]
unsafe extern "system" {
    fn RtlGenRandom(random_buffer: *mut u8, length: u32) -> BOOLEAN;
}

#[allow(clippy::upper_case_acronyms)]
type BOOLEAN = i32;
const FALSE: BOOLEAN = 0;
