use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    let _len = u64::try_from(bytes.len()).unwrap();
    #[cfg(target_env = "p2")]
    { bytes.copy_from_slice(&get_random_bytes(len)); }
    #[cfg(target_env = "p3")]
    { bytes.copy_from_slice(&get_random_bytes(len)); }
    Ok(())
}

pub fn hashmap_random_keys() -> Result<(u64, u64), RandomError> {
    #[cfg(target_env = "p2")]
    return Ok(get_insecure_seed());
    #[cfg(target_env = "p3")]
    return Ok(get_insecure_seed());
    // Unreachable: this module is only compiled for wasi p2/p3.
    unreachable!()
}

unsafe extern "C" {
    fn random_get_random_bytes(len: u64) -> *const u8;
    fn random_free(buf: *const u8, len: usize);
}

fn get_random_bytes(len: u64) -> Vec<u8> {
    unsafe {
        let ptr = random_get_random_bytes(len);
        let v = std::slice::from_raw_parts(ptr, len as usize).to_vec();
        random_free(ptr, len as usize);
        v
    }
}

unsafe extern "C" {
    fn insecure_seed_insecure_seed(k1: *mut u64, k2: *mut u64);
    fn insecure_seed_get_insecure_seed(k1: *mut u64, k2: *mut u64);
}

#[cfg(target_env = "p2")]
fn get_insecure_seed() -> (u64, u64) {
    unsafe {
        let mut k1: u64 = 0;
        let mut k2: u64 = 0;
        insecure_seed_insecure_seed(&mut k1, &mut k2);
        (k1, k2)
    }
}

#[cfg(target_env = "p3")]
fn get_insecure_seed() -> (u64, u64) {
    unsafe {
        let mut k1: u64 = 0;
        let mut k2: u64 = 0;
        insecure_seed_get_insecure_seed(&mut k1, &mut k2);
        (k1, k2)
    }
}
