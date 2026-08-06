// Module verified
use super::RandomError;

const RETRIES: u32 = 10;

fn rdrand64() -> Result<u64, RandomError> {
    unsafe {
        let mut ret: u64 = 0;
        for _ in 0..RETRIES {
            if core::arch::x86_64::_rdrand64_step(&mut ret) == 1 {
                return Ok(ret);
            }
        }
        Err(RandomError::Platform(core::borrow::Cow::Borrowed(
            "RDRAND64 failed after retries",
        )))
    }
}

fn rdrand32() -> Result<u32, RandomError> {
    unsafe {
        let mut ret: u32 = 0;
        for _ in 0..RETRIES {
            if core::arch::x86_64::_rdrand32_step(&mut ret) == 1 {
                return Ok(ret);
            }
        }
        Err(RandomError::Platform(core::borrow::Cow::Borrowed(
            "RDRAND32 failed after retries",
        )))
    }
}

fn rdrand16() -> Result<u16, RandomError> {
    unsafe {
        let mut ret: u16 = 0;
        for _ in 0..RETRIES {
            if core::arch::x86_64::_rdrand16_step(&mut ret) == 1 {
                return Ok(ret);
            }
        }
        Err(RandomError::Platform(core::borrow::Cow::Borrowed(
            "RDRAND16 failed after retries",
        )))
    }
}

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    let (chunks, remainder) = bytes.as_chunks_mut();
    for chunk in chunks {
        *chunk = rdrand64()?.to_ne_bytes();
    }

    let (chunks, remainder) = remainder.as_chunks_mut();
    for chunk in chunks {
        *chunk = rdrand32()?.to_ne_bytes();
    }

    let (chunks, remainder) = remainder.as_chunks_mut();
    for chunk in chunks {
        *chunk = rdrand16()?.to_ne_bytes();
    }

    if let [byte] = remainder {
        *byte = rdrand16()? as u8;
    }
    Ok(())
}
