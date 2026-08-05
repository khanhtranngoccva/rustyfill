//! Contains most of the shared UEFI specific stuff. Some of this might be moved to `std::os::uefi`
//! if needed but no point in adding extra public API when there is not Std support for UEFI in the
//! first place
//!
//! Some Nomenclature
//! * Protocol:
//! - Protocols serve to enable communication between separately built modules, including drivers.
//! - Every protocol has a GUID associated with it. The GUID serves as the name for the protocol.
//! - Protocols are produced and consumed.
//! - More information about protocols can be found [here](https://edk2-docs.gitbook.io/edk-ii-uefi-driver-writer-s-guide/3_foundation/36_protocols_and_handles)

use r_efi::efi::{self, Guid};
use r_efi::protocols::{device_path, device_path_to_text, file, service_binding, shell};

use std::alloc::Layout;
use std::ffi::{OsStr, OsString};
use std::io::{self, const_error};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::os::uefi::env::boot_services;
use std::os::uefi::ffi::{OsStrExt, OsStringExt};
use std::os::uefi::{self};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{Atomic, AtomicPtr, Ordering};

type BootInstallMultipleProtocolInterfaces =
    unsafe extern "efiapi" fn(_: *mut r_efi::efi::Handle, _: ...) -> r_efi::efi::Status;

type BootUninstallMultipleProtocolInterfaces =
    unsafe extern "efiapi" fn(_: r_efi::efi::Handle, _: ...) -> r_efi::efi::Status;

const BOOT_SERVICES_UNAVAILABLE: io::Error = const_error!(
    io::ErrorKind::Other,
    "Boot Services are no longer available"
);
const OUT_OF_MEMORY: io::Error = const_error!(io::ErrorKind::OutOfMemory, "Out of memory");

/// Locates Handles with a particular Protocol GUID.
///
/// Implemented using `EFI_BOOT_SERVICES.LocateHandles()`.
///
/// Returns an array of [Handles](r_efi::efi::Handle) that support a specified protocol.
pub(crate) fn locate_handles(mut guid: Guid) -> io::Result<Vec<NonNull<crate::ffi::c_void>>> {
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
                crate::ptr::null_mut(),
                buf_size,
                buf,
            )
        };

        if r.is_error() {
            Err(crate::io::Error::from_raw_os_error(r.as_usize()))
        } else {
            Ok(())
        }
    }

    let boot_services = boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let mut buf_len = 0usize;

    // This should always fail since the size of buffer is 0. This call should update the buf_len
    // variable with the required buffer length
    match inner(
        &mut guid,
        boot_services,
        &mut buf_len,
        crate::ptr::null_mut(),
    ) {
        Ok(()) => unreachable!(),
        Err(e) => match e.kind() {
            io::ErrorKind::FileTooLarge => {}
            _ => return Err(e),
        },
    }

    // The returned buf_len is in bytes
    assert_eq!(buf_len % size_of::<r_efi::efi::Handle>(), 0);
    let num_of_handles = buf_len / size_of::<r_efi::efi::Handle>();
    let mut buf: Vec<r_efi::efi::Handle> = Vec::fallible_(num_of_handles);
    match inner(&mut guid, boot_services, &mut buf_len, buf.as_mut_ptr()) {
        Ok(()) => {
            // This is safe because the call will succeed only if buf_len >= required length.
            // Also, on success, the `buf_len` is updated with the size of bufferv (in bytes) written
            unsafe { buf.set_len(num_of_handles) };
            Ok(buf.into_iter().filter_map(|x| NonNull::new(x)).collect())
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
pub(crate) fn open_protocol<T>(
    handle: NonNull<crate::ffi::c_void>,
    mut protocol_guid: Guid,
) -> io::Result<NonNull<T>> {
    let boot_services: NonNull<efi::BootServices> =
        boot_services().ok_or(BOOT_SERVICES_UNAVAILABLE)?.cast();
    let system_handle = uefi::env::image_handle();
    let mut protocol: MaybeUninit<*mut T> = MaybeUninit::uninit();

    let r = unsafe {
        ((*boot_services.as_ptr()).open_protocol)(
            handle.as_ptr(),
            &mut protocol_guid,
            protocol.as_mut_ptr().cast(),
            system_handle.as_ptr(),
            crate::ptr::null_mut(),
            r_efi::system::OPEN_PROTOCOL_GET_PROTOCOL,
        )
    };

    if r.is_error() {
        Err(crate::io::Error::from_raw_os_error(r.as_usize()))
    } else {
        NonNull::new(unsafe { protocol.assume_init() })
            .ok_or(const_error!(io::ErrorKind::Other, "null protocol"))
    }
}
