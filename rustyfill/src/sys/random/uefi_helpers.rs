//! Partial port for the PAL module, sufficient for dealing with random. There are sharp edges!
#![allow(unused)]

use crate::vec::TryVec;
use r_efi::efi::{self, Guid, Status};
use std::io::{self};
use std::mem::MaybeUninit;

#[rustversion::nightly]
use std::os::uefi::env::{boot_services, image_handle};
#[rustversion::nightly]
use std::io::const_error;

use std::ptr::{self, NonNull};

#[rustversion::nightly]
const BOOT_SERVICES_UNAVAILABLE: io::Error = const_error!(
    io::ErrorKind::Other,
    "Boot Services are no longer available"
);
#[rustversion::nightly]
const OUT_OF_MEMORY: io::Error = const_error!(io::ErrorKind::Other, "Out of memory");

/// Locates Handles with a particular Protocol GUID.
///
/// Implemented using `EFI_BOOT_SERVICES.LocateHandles()`.
///
/// Returns an array of [Handles](r_efi::efi::Handle) that support a specified protocol.
#[rustversion::nightly]
pub(crate) fn locate_handles(mut guid: Guid) -> io::Result<Vec<NonNull<core::ffi::c_void>>> {
    fn inner(
        guid: &mut Guid,
        boot_services: NonNull<r_efi::efi::BootServices>,
        buf_size: &mut usize,
        buf: *mut r_efi::efi::Handle,
    ) -> io::Result<()> {
        let r = unsafe {
            ((*boot_services.as_ptr()).locate_handle)(
                r_efi::efi::BY_PROTOCOL,
                guid,
                ptr::null_mut(),
                buf_size,
                buf,
            )
        };

        if r.is_error() {
            Err(io::Error::from_raw_os_error(r.as_usize()))
        } else {
            Ok(())
        }
    }

    // Out of boot services, mark as unsupported
    let boot_services = boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let mut buf_len = 0usize;

    // This should always fail since the size of buffer is 0. This call should update the buf_len
    // variable with the required buffer length
    match inner(&mut guid, boot_services, &mut buf_len, ptr::null_mut()) {
        Ok(()) => unreachable!(),
        Err(e) => match e.kind() {
            io::ErrorKind::FileTooLarge => {}
            _ => return Err(e),
        },
    }

    // The returned buf_len is in bytes
    assert_eq!(buf_len % size_of::<r_efi::efi::Handle>(), 0);
    let num_of_handles = buf_len / size_of::<r_efi::efi::Handle>();
    let mut buf: Vec<r_efi::efi::Handle> =
        Vec::fallible_with_capacity(num_of_handles).map_err(|_| OUT_OF_MEMORY)?;
    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            // This is safe because the call will succeed only if buf_len >= required length.
            // Also, on success, the `buf_len` is updated with the size of bufferv (in bytes) written
            unsafe { buf.set_len(num_of_handles) };
            let items = Vec::fallible_collect(buf.into_iter().filter_map(|x| NonNull::new(x)))
                .map_err(|_| OUT_OF_MEMORY)?;
            Ok(items)
        }
        Err(e) => Err(e),
    }
}

/// Open Protocol on a handle.
/// Internally just a call to `EFI_BOOT_SERVICES.OpenProtocol()`.
///
/// Queries a handle to determine if it supports a specified protocol. If the protocol is
/// supported by the handle, it opens the protocol on behalf of the calling agent.
///
/// The protocol is opened with the attribute GET_PROTOCOL, which means the caller is not required
/// to close the protocol interface with `EFI_BOOT_SERVICES.CloseProtocol()`
#[rustversion::nightly]
pub(crate) fn open_protocol<T>(
    handle: NonNull<core::ffi::c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services: NonNull<efi::BootServices> =
        boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let system_handle = image_handle();
    let mut protocol: MaybeUninit<*mut T> = MaybeUninit::uninit();

    let r = unsafe {
        ((*boot_services.as_ptr()).open_protocol)(
            handle.as_ptr(),
            &mut protocol_guid,
            protocol.as_mut_ptr().cast(),
            system_handle.as_ptr(),
            ptr::null_mut(),
            r_efi::system::OPEN_PROTOCOL_GET_PROTOCOL,
        )
    };

    if r.is_error() {
        Err(io::Error::from_raw_os_error(r.as_usize()))
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(const_error!(io::ErrorKind::Other, "null protocol"))
    }
}
