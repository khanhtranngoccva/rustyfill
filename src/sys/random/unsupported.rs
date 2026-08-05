use super::RandomError;

pub fn fill_bytes(_: &mut [u8]) -> Result<(), RandomError> {
    Err(RandomError::Unsupported)
}

pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    // Use allocation addresses for a bit of randomness. This isn't
    // particularly secure, but there isn't really an alternative.
    let stack = 0u8;
    let heap = Box::new(0u8);
    let k1 = std::ptr::from_ref(&stack).addr() as u64;
    let k2 = std::ptr::from_ref(&*heap).addr() as u64;
    Ok((k1, k2))
}
