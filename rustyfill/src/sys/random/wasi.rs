use super::RandomError;

#[cfg(target_env = "p2")]
use wasip2::random::{insecure_seed::insecure_seed as get_insecure_seed, random::get_random_bytes};
#[cfg(target_env = "p3")]
use wasip3::random::{insecure_seed::get_insecure_seed, random::get_random_bytes};

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    bytes.copy_from_slice(&get_random_bytes(u64::try_from(bytes.len()).unwrap()));
    Ok(())
}

pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    Ok(get_insecure_seed())
}
