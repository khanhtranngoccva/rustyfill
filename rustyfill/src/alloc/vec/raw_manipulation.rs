use crate::alloc::AllocError;
use lang_alloc::vec::Vec;
use lang_core::alloc::Layout;
use lang_core::ptr::{self, NonNull};

/// # Safety
///
/// `cap` must be in the `0..=isize::MAX` range.
type Cap = usize;

/// # A raw view of a vector that is allocated using the global allocator.
#[allow(missing_debug_implementations)]
pub(crate) struct RawVecInnerView {
    ptr: NonNull<u8>,
    cap: Cap,
}

impl RawVecInnerView {
    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    #[inline]
    const unsafe fn current_memory(&self, elem_layout: Layout) -> Option<(NonNull<u8>, Layout)> {
        if elem_layout.size() == 0 || self.cap == 0 {
            None
        } else {
            // We could use Layout::array here which ensures the absence of isize and usize overflows
            // and could hypothetically handle differences between stride and size, but this memory
            // has already been allocated so we know it can't overflow and currently Rust does not
            // support such types. So we can do better by skipping some checks and avoid an unwrap.
            unsafe {
                let alloc_size = elem_layout.size().unchecked_mul(self.cap);
                let layout = Layout::from_size_align_unchecked(alloc_size, elem_layout.align());
                Some((self.ptr, layout))
            }
        }
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - `cap` must be less than or equal to `self.capacity(elem_layout.size())`
    /// - `cap <= self.cap && cap <= isize::MAX`
    #[cfg_attr(test, no_panic::no_panic)]
    pub(crate) unsafe fn shrink_unchecked(
        &mut self,
        cap: usize,
        elem_layout: Layout,
    ) -> Result<(), AllocError> {
        // SAFETY: Precondition passed to caller
        let Some((ptr, layout)) = (unsafe { self.current_memory(elem_layout) }) else {
            return Ok(());
        };

        // If shrinking to 0, deallocate the buffer. We don't reach this point
        // for the T::IS_ZST case since current_memory() will have returned
        // None.
        if cap == 0 {
            unsafe {
                ::lang_alloc::alloc::dealloc(ptr.as_ptr(), layout);
            }
            self.ptr = NonNull::new(ptr::without_provenance_mut(elem_layout.align()))
                .expect("alignment should not be zero");
            self.cap = 0;
        } else {
            let ptr = unsafe {
                // Layout cannot overflow here because it would have
                // overflowed earlier when capacity was larger.
                // new_size > 0 because both cap and size must be positive.
                // Alignment must match, so only the realloc branch is taken
                let new_size = elem_layout.size().unchecked_mul(cap);
                let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
                // SAFETY: new_layout.align() == elem_layout.align()
                NonNull::new(::lang_alloc::alloc::realloc(ptr.as_ptr(), layout, new_size))
                    .ok_or(AllocError { layout: new_layout })?
            };
            // SAFETY: if the allocation is valid, then the capacity is too
            self.ptr = ptr;
            self.cap = cap;
        }
        Ok(())
    }

    #[cfg_attr(test, no_panic::no_panic)]
    pub(crate) fn from_vec<T>(vec: Vec<T>) -> (Self, usize) {
        let (ptr, len, cap) = vec.into_raw_parts();
        (
            RawVecInnerView {
                ptr: unsafe { NonNull::new_unchecked(ptr.cast()) },
                cap,
            },
            len,
        )
    }

    #[cfg_attr(test, no_panic::no_panic)]
    pub(crate) unsafe fn into_vec<T>(self, len: usize) -> Vec<T> {
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr().cast(), len, self.cap) }
    }
}
