// Module verified
use std::fs::File;
use std::io::Read;
// Fallback for get_or_try_init API
use once_cell::sync::OnceCell;

use super::RandomError;

static SCHEME: OnceCell<File> = OnceCell::new();

pub fn fill_bytes(bytes: &mut [u8]) -> Result<(), RandomError> {
    SCHEME
        .get_or_try_init(|| File::open("/scheme/rand"))
        .and_then(|mut scheme| scheme.read_exact(bytes))
        .map_err(|e| {
            RandomError::Platform(format!("failed to open and read /scheme/rand: {}", e))
        })?;
    Ok(())
}
