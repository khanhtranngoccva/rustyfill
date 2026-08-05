use super::RandomError;

pub fn fill_bytes(_: &mut [u8]) -> Result<(), RandomError> {
    Err(RandomError::Unsupported)
}

pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    // No OS random source available; fall back to a stack-based Mersenne Twister
    // seeded from the stack address. Not cryptographically secure, but sufficient
    // for hashmap seed diversity.
    Ok(super::hashmap_random_keys_mt())
}
