// Module verified
use super::RandomError;

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    moto_rt::fill_random_bytes(bytes);
    Ok(())
}
