//! A fallible string interner backed by a static [`ConcurrentHashMap`].
//!
//! The interner deduplicates values at runtime using a global concurrent hash map.
//! When you call [`intern()`](InternExt::intern) on a borrowed value (e.g., `&str`),
//! the interner checks if an identical value already exists. If it does, you get back
//! a shared reference to the existing copy. If not, the value is cloned into an
//! [`Arc`] and stored globally.
//!
//! # Pruning
//!
//! Every `PRUNE_INTERVAL` intern calls, unlocked shards are scanned for expired
//! [`Weak`] references and those entries are removed, reclaiming memory.
//!
//! # Usage
//!
//! ```
//! use rustyfill::collections::interner::{Intern, InternExt};
//!
//! let a = "hello".intern().unwrap();
//! let b = "hello".intern().unwrap();
//! assert!(a.ptr_eq(&b)); // Same underlying Arc
//! ```

use crate::alloc::TryReserveErrorExt;
use crate::collections::chashmap::ConcurrentHashMap;
use crate::std::arc::{TryArc, TryWeak};
use crate::try_clone::TryClone;
use crate::try_clone::TryCloneError;
use crate::try_default::{TryDefault, TryDefaultError};
use crate::try_fmt::FormatterExt;
use crate::try_fmt::TryDebug;
use crate::try_to_owned::{TryToOwned, TryToOwnedError};
use lang_alloc::borrow::ToOwned;
use lang_alloc::string::String;
use lang_core::alloc::Layout;
use lang_core::fmt;
use lang_core::hash::Hash;
use lang_std::ffi::{CStr, CString, OsStr, OsString};
use lang_std::hash::{BuildHasher as _, RandomState};
use lang_std::ops::Deref;
use lang_std::path::{Path, PathBuf};
use lang_std::sync::atomic::{AtomicUsize, Ordering};
use lang_std::sync::{Arc, Weak};

/// Number of intern calls between pruning sweeps of unlocked shards.
const PRUNE_INTERVAL: usize = 1024;

// ── Intern enum ────────────────────────────────────────────────────────────────

/// Sealed trait mapping a borrowed type to its owned counterpart.
///
/// Implemented only for `str`, `OsStr`, `CStr`, and `Path`.
/// Mirrors the `ToOwned` relationship with stronger fallible
/// semantics: the borrowed type must also implement [`TryToOwned`], and the
/// owned type must be clonable so we can build the Arc before acquiring locks.
pub trait InternKind: TryToOwned + Hash + Eq + 'static
where
    <Self as ToOwned>::Owned: Hash + Eq + PartialEq<Self> + TryClone,
{
}

impl InternKind for str {}

impl InternKind for OsStr {}

impl InternKind for CStr {}

impl InternKind for Path {}

/// An interned value that may be owned or shared via Arc.
///
/// The type parameter `B` is the borrowed form (`str`, `OsStr`, `CStr`, `Path`).
/// The owned form is provided by `<B as ToOwned>::Owned`.
///
/// Supported types: `Intern<str>`, `Intern<OsStr>`, `Intern<CStr>`, `Intern<Path>`.
pub enum Intern<B: InternKind + ?Sized>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    /// An owned value, not yet interned.
    Owned(<B as ToOwned>::Owned),
    /// A shared reference to a globally-interned value.
    Shared(Arc<<B as ToOwned>::Owned>),
}

// Type aliases for ergonomics
pub type InternStr = Intern<str>;
pub type InternOsStr = Intern<OsStr>;
pub type InternCStr = Intern<CStr>;
pub type InternPath = Intern<Path>;

impl<B: InternKind + ?Sized> Clone for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Owned(s) => Self::Owned(s.clone()),
            Self::Shared(a) => Self::Shared(a.clone()),
        }
    }
}

impl<B: InternKind + ?Sized> TryClone for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    fn try_clone(&self) -> Result<Self, TryCloneError> {
        Ok(match self {
            Self::Owned(s) => Self::Owned(s.try_clone()?),
            Self::Shared(a) => Self::Shared(a.try_clone()?),
        })
    }
}

impl<B: InternKind + ?Sized> From<Arc<<B as ToOwned>::Owned>> for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    fn from(value: Arc<<B as ToOwned>::Owned>) -> Self {
        Self::Shared(value)
    }
}

// Cannot implement From<<B as ToOwned>::Owned> for Intern<B> because it conflicts
// with the blanket impl<T> From<T> for T in core (coherence error E0119): the
// compiler cannot prove that Intern<B> and <B as ToOwned>::Owned are distinct types
// for all possible B. Use Intern::from_owned() instead.

// Common impls for all Intern variants
impl<B: InternKind + ?Sized> Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    /// Returns true if this is the `Shared` variant.
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    /// Returns true if this is the `Owned` variant.
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Creates an owned variant from an owned value, bypassing the interner.
    pub fn from_owned(owned: <B as ToOwned>::Owned) -> Self {
        Self::Owned(owned)
    }

    /// Checks pointer equality between two `Intern` values.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Shared(a), Self::Shared(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl<B: InternKind + ?Sized> PartialEq for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(a), Self::Owned(b)) => a == b,
            (Self::Shared(a), Self::Shared(b)) => Arc::ptr_eq(a, b) || (**a) == (**b),
            (Self::Owned(a), Self::Shared(b)) => a == b.as_ref(),
            (Self::Shared(a), Self::Owned(b)) => a.as_ref() == b,
        }
    }
}

impl<B: InternKind + ?Sized> Eq for Intern<B> where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone
{
}

impl<B: InternKind + ?Sized> Deref for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + Deref<Target = B>,
{
    type Target = B;

    fn deref(&self) -> &B {
        match self {
            Self::Owned(o) => o.deref(),
            Self::Shared(a) => a.deref(),
        }
    }
}

impl<B: InternKind + ?Sized> AsRef<B> for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + AsRef<B>,
{
    fn as_ref(&self) -> &B {
        match self {
            Self::Owned(o) => o.as_ref(),
            Self::Shared(a) => a.as_ref().as_ref(),
        }
    }
}

impl<B: InternKind + ?Sized> fmt::Debug for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(v) => f.debug_tuple("Owned").field(v).finish(),
            Self::Shared(v) => f.debug_tuple("Shared").field(v).finish(),
        }
    }
}

impl<B: InternKind + ?Sized> TryDebug for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + TryDebug,
{
    fn try_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(v) => f.try_debug_tuple("Owned").field(v).finish(),
            Self::Shared(v) => f.try_debug_tuple("Shared").field(v).finish(),
        }
    }
}

impl<B: InternKind + ?Sized> Default for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + Default,
{
    fn default() -> Self {
        Intern::Owned(Default::default())
    }
}

impl<B: InternKind + ?Sized> TryDefault for Intern<B>
where
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + TryClone + TryDefault,
{
    fn try_default() -> Result<Self, TryDefaultError> {
        Ok(Intern::Owned(TryDefault::try_default()?))
    }
}

// ── Global interner state ──────────────────────────────────────────────────────
//
// Key:   u64 — precomputed hash of the borrowed value. Standard Hash/Eq on u64,
//        no stubs needed. Actual value equality is checked in find() closures
//        by upgrading the Weak and comparing against the borrowed input.
// Value: Weak<Owned> — does not keep the Arc alive. When all external
//        Intern::Shared handles drop, the Arc is freed and the Weak expires,
//        allowing the entry to be pruned.

crate::collections::chashmap::declare_concurrent_hash_map! {
    pub(crate) static INTERNER_STR: ConcurrentHashMap<u64, lang_std::sync::Weak<lang_alloc::string::String>> = 64
}

crate::collections::chashmap::declare_concurrent_hash_map! {
    pub(crate) static INTERNER_OS_STR: ConcurrentHashMap<u64, lang_std::sync::Weak<lang_std::ffi::OsString>> = 64
}

crate::collections::chashmap::declare_concurrent_hash_map! {
    pub(crate) static INTERNER_CSTR: ConcurrentHashMap<u64, lang_std::sync::Weak<lang_std::ffi::CString>> = 64
}

crate::collections::chashmap::declare_concurrent_hash_map! {
    pub(crate) static INTERNER_PATH: ConcurrentHashMap<u64, lang_std::sync::Weak<lang_std::path::PathBuf>> = 64
}

/// Global monotonic counter tracking total intern calls across all types.
/// Used to trigger periodic pruning every `PRUNE_INTERVAL` calls.
static INTERN_CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Prune unlocked shards across all four interner maps if sufficient interning operations are reached.
fn maybe_prune_all_maps() {
    let prev_total = INTERN_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    if !prev_total.is_multiple_of(PRUNE_INTERVAL) {
        return;
    }

    prune_unlocked_shards(&*INTERNER_STR);
    prune_unlocked_shards(&*INTERNER_OS_STR);
    prune_unlocked_shards(&*INTERNER_CSTR);
    prune_unlocked_shards(&*INTERNER_PATH);
}

/// Scan unlocked shards of a map and remove entries whose `Weak` has expired.
///
/// When all external [`Intern::Shared`] handles are dropped, the last strong ref
/// to the `Arc<Owned>` is released. The `Weak` value can no longer upgrade.
///
/// Uses `try_write_table` so we never block waiting for a contended shard.
fn prune_unlocked_shards<Owned>(map: &ConcurrentHashMap<u64, Weak<Owned>, RandomState>)
where
    Owned: 'static,
{
    let shards = map.get_shards();
    for shard in shards {
        let mut guard = match shard.try_write_table() {
            Some(g) => g,
            None => continue,
        };

        let i = unsafe { guard.iter() };
        for bucket in i {
            let weak = unsafe { &bucket.as_ref().1 };
            // Use try_upgrade for an atomic check-and-increment: if this succeeds,
            // a strong reference still exists (or just came back), so we keep the
            // entry and drop the temporary Arc. If it returns None, the Arc has
            // been fully dropped and the entry is stale. Some(Err) means refcount
            // overflow — also treat as stale since we can't verify liveness.
            match weak.try_upgrade() {
                Some(Ok(_arc)) => { /* still alive */ }
                None | Some(Err(_)) => unsafe {
                    let _removed = guard.remove(bucket);
                },
            }
        }
    }
}

/// Generic intern logic shared by all interner backends.
///
/// Accepts a borrowed value (`&B`). Uses [`TryToOwned::try_to_owned`] to lazily
/// construct the owned value (only on cache miss), then [`TryArc::fallible_new`]
/// to allocate the `Arc`. All hashing, allocation, locking, and comparison happen
/// inside this function.
///
/// The backing map stores `u64 -> Weak<Owned>`. The precomputed hash of the
/// borrowed value serves as both the map key and the RawTable probe hash. Value
/// equality is checked in `find()` closures by upgrading the `Weak` and comparing
/// the resulting `Arc` against the borrowed input.
///
/// # Return
/// - `Ok(Arc<Owned>)` on success
/// - `Err(e)` if `try_to_owned` failed for any reason
fn do_intern_with_key<B>(
    map: &ConcurrentHashMap<u64, Weak<B::Owned>, RandomState>,
    borrowed: &B,
) -> Result<Arc<B::Owned>, TryToOwnedError>
where
    B: InternKind + ?Sized,
    <B as ToOwned>::Owned: Hash + Eq + PartialEq<B> + Clone + TryClone,
{
    // ── Compute hash (infallible, no locks) ──────────────────────────────────
    let hash = map.compute_hash_internal(borrowed);

    // ── Step 1: fast path — read lock, check cache ──────────────────────────
    // No owned value constructed yet. If we find a hit, try_to_owned is never called.
    {
        let idx = map.shard_index_internal(hash);
        let shard = &map.get_shards()[idx];
        let guard = shard.read_table();

        if let Some(bucket) = guard.find(hash, |kv: &(u64, Weak<B::Owned>)| {
            let (_, weak) = kv;
            let inner: Arc<B::Owned> = match weak.try_upgrade() {
                Some(Err(_)) | None => return false,
                Some(Ok(s)) => s,
            };
            inner.as_ref() == borrowed
        }) {
            let weak = unsafe { &bucket.as_ref().1 };
            if let Some(Ok(arc)) = weak.try_upgrade() {
                return Ok(arc);
            }
        }
    }

    // ── Step 2: slow path — write lock, reserve, insert ─────────────────────
    let idx = map.shard_index_internal(hash);
    let shard = &map.get_shards()[idx];
    let mut guard = shard.write_table();

    if guard
        .try_reserve(1, |(k, _v): &(u64, Weak<B::Owned>)| {
            map.hasher().hash_one(k)
        })
        .is_err()
    {
        return Err(TryToOwnedError::Reserve(TryReserveErrorExt::new_alloc(
            Layout::new::<u8>(),
        )));
    }

    let create_new_arc = || -> Result<Arc<B::Owned>, TryToOwnedError> {
        let owned = borrowed.try_to_owned()?;
        let arc = match Arc::fallible_new(owned) {
            Ok(a) => a,
            Err(alloc_err) => return Err(TryToOwnedError::Alloc(alloc_err)),
        };
        Ok(arc)
    };

    // Re-check under write lock: another thread may have inserted between
    // our read-check dropping and this write lock acquiring.
    if let Some(bucket) = guard.find(hash, |kv: &(u64, Weak<B::Owned>)| {
        let (_, weak) = kv;
        let inner: Arc<B::Owned> = match weak.try_upgrade() {
            Some(Err(_)) | None => return false,
            Some(Ok(s)) => s,
        };
        inner.as_ref() == borrowed
    }) {
        let weak = unsafe { &bucket.as_ref().1 };
        if let Some(Ok(upgraded)) = weak.try_upgrade() {
            return Ok(upgraded);
        }
        let arc = create_new_arc()?;
        // Downgrading here cannot cause overflow.
        unsafe { bucket.as_mut().1 = Arc::downgrade(&arc) };
        return Ok(arc);
    }

    let arc = create_new_arc()?;
    // Truly new — insert. Downgrading here cannot cause overflow.
    let weak: Weak<B::Owned> = Arc::downgrade(&arc);

    // SAFETY: We already reserved space above with try_reserve(1).
    unsafe {
        guard.insert_no_grow(hash, (hash, weak));
    }

    Ok(arc)
}

// ── InternExt trait ────────────────────────────────────────────────────────────

/// Extension methods for internable borrowed types.
pub trait InternExt: InternKind
where
    <Self as ToOwned>::Owned: Hash + Eq + PartialEq<Self> + TryClone,
{
    /// Fallibly intern this value via the global interner.
    fn intern(&self) -> Result<Intern<Self>, TryToOwnedError>;

    /// Like [`Self::intern`] but takes ownership of the owned value.
    fn intern_owned(owned: Self::Owned) -> Result<Intern<Self>, TryToOwnedError>;
}

// ── str implementation ─────────────────────────────────────────────────────────

impl InternExt for str {
    fn intern(&self) -> Result<Intern<Self>, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<str>(&*INTERNER_STR, self).map(InternStr::from)
    }

    fn intern_owned(owned: String) -> Result<Intern<Self>, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<str>(&*INTERNER_STR, &owned).map(InternStr::from)
    }
}

impl InternStr {
    /// Attempt to intern this value. If `Owned`, tries to find/create a shared Arc.
    /// On success returns `Shared`, on failure returns `Owned` with the original value.
    pub fn promote(self) -> Self {
        match self {
            Self::Shared(_) => self,
            Self::Owned(owned) => {
                maybe_prune_all_maps();
                match do_intern_with_key::<str>(&*INTERNER_STR, &owned) {
                    Ok(arc) => Self::Shared(arc),
                    Err(_) => Self::Owned(owned),
                }
            }
        }
    }
}

impl PartialEq<String> for InternStr {
    fn eq(&self, other: &String) -> bool {
        match self {
            Self::Owned(a) => a == other,
            Self::Shared(a) => a.as_ref() == other.as_str(),
        }
    }
}

impl PartialEq<str> for InternStr {
    fn eq(&self, other: &str) -> bool {
        match self {
            Self::Owned(a) => a.as_str() == other,
            Self::Shared(a) => a.as_ref() == other,
        }
    }
}

impl PartialEq<&str> for InternStr {
    fn eq(&self, other: &&str) -> bool {
        PartialEq::eq(self, *other)
    }
}

impl fmt::Display for InternStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.deref())
    }
}

// ── OsStr implementation ───────────────────────────────────────────────────────

impl InternExt for OsStr {
    fn intern(&self) -> Result<InternOsStr, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<OsStr>(&*INTERNER_OS_STR, self).map(InternOsStr::from)
    }

    fn intern_owned(owned: OsString) -> Result<InternOsStr, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<OsStr>(&*INTERNER_OS_STR, &owned).map(InternOsStr::from)
    }
}

impl InternOsStr {
    pub fn promote(self) -> Self {
        match self {
            Self::Shared(_) => self,
            Self::Owned(owned) => {
                maybe_prune_all_maps();
                match do_intern_with_key::<OsStr>(&*INTERNER_OS_STR, &owned) {
                    Ok(arc) => Self::Shared(arc),
                    Err(_) => Self::Owned(owned),
                }
            }
        }
    }
}

impl PartialEq<OsString> for InternOsStr {
    fn eq(&self, other: &OsString) -> bool {
        match self {
            Self::Owned(a) => a == other,
            Self::Shared(a) => a.as_ref() == other.as_os_str(),
        }
    }
}

impl PartialEq<OsStr> for InternOsStr {
    fn eq(&self, other: &OsStr) -> bool {
        match self {
            Self::Owned(a) => a.as_os_str() == other,
            Self::Shared(a) => a.as_ref() == other,
        }
    }
}

impl fmt::Display for InternOsStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.deref().to_string_lossy())
    }
}

// ── CStr implementation ────────────────────────────────────────────────────────

impl InternExt for CStr {
    fn intern(&self) -> Result<InternCStr, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<CStr>(&*INTERNER_CSTR, self).map(InternCStr::from)
    }

    fn intern_owned(owned: CString) -> Result<InternCStr, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<CStr>(&*INTERNER_CSTR, &owned).map(InternCStr::from)
    }
}

impl InternCStr {
    pub fn promote(self) -> Self {
        match self {
            Self::Shared(_) => self,
            Self::Owned(owned) => {
                maybe_prune_all_maps();
                match do_intern_with_key::<CStr>(&*INTERNER_CSTR, &owned) {
                    Ok(arc) => Self::Shared(arc),
                    Err(_) => Self::Owned(owned),
                }
            }
        }
    }
}

impl PartialEq<CStr> for InternCStr {
    fn eq(&self, other: &CStr) -> bool {
        match self {
            Self::Owned(a) => a.as_c_str() == other,
            Self::Shared(a) => a.as_ref() == other,
        }
    }
}

impl fmt::Display for InternCStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.deref().to_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid utf-8 cstr>"),
        }
    }
}

// ── Path implementation ────────────────────────────────────────────────────────

impl InternExt for Path {
    fn intern(&self) -> Result<InternPath, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<Path>(&*INTERNER_PATH, self).map(InternPath::from)
    }

    fn intern_owned(owned: PathBuf) -> Result<InternPath, TryToOwnedError> {
        maybe_prune_all_maps();
        do_intern_with_key::<Path>(&*INTERNER_PATH, &owned).map(InternPath::from)
    }
}

impl InternPath {
    pub fn promote(self) -> Self {
        match self {
            Self::Shared(_) => self,
            Self::Owned(owned) => {
                maybe_prune_all_maps();
                match do_intern_with_key::<Path>(&*INTERNER_PATH, &owned) {
                    Ok(arc) => Self::Shared(arc),
                    Err(_) => Self::Owned(owned),
                }
            }
        }
    }
}

impl PartialEq<PathBuf> for InternPath {
    fn eq(&self, other: &PathBuf) -> bool {
        match self {
            Self::Owned(a) => a == other,
            Self::Shared(a) => a.as_ref() == other.as_path(),
        }
    }
}

impl PartialEq<Path> for InternPath {
    fn eq(&self, other: &Path) -> bool {
        match self {
            Self::Owned(a) => a.as_path() == other,
            Self::Shared(a) => a.as_ref() == other,
        }
    }
}

impl fmt::Display for InternPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.deref().display())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lang_alloc::string::String;
    use lang_alloc::string::ToString;
    use lang_alloc::vec::Vec;
    use lang_std::format;

    #[test]
    fn intern_basic() {
        let a = "hello".intern().unwrap();
        assert_eq!(&*a, "hello");
        assert!(a.is_shared());
    }

    #[test]
    fn intern_deduplication() {
        let a = "world".intern().unwrap();
        let b = "world".intern().unwrap();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn intern_different_values() {
        let a = "foo".intern().unwrap();
        let b = "bar".intern().unwrap();
        assert!(!a.ptr_eq(&b));
        assert_eq!(&*a, "foo");
        assert_eq!(&*b, "bar");
    }

    #[test]
    fn intern_display() {
        let s = "display test".intern().unwrap();
        let formatted = format!("{}", s);
        assert_eq!(formatted, "display test");
    }

    #[test]
    fn intern_as_ref() {
        let s: InternStr = "as ref test".intern().unwrap();
        let slice: &str = s.as_ref();
        assert_eq!(slice, "as ref test");
    }

    #[test]
    fn intern_owned_variant() {
        let owned = Intern::<str>::from_owned("owned".to_string());
        assert!(owned.is_owned());
        assert_eq!(&*owned, "owned");
    }

    #[test]
    fn intern_equality_cross_variant() {
        let shared = "equality".intern().unwrap();
        let owned = Intern::<str>::from_owned("equality".to_string());
        assert_eq!(shared, owned);
    }

    #[test]
    fn intern_equality_against_string() {
        let shared = "compare".intern().unwrap();
        assert_eq!(shared, String::from("compare"));
        assert_ne!(shared, String::from("different"));
    }

    #[test]
    fn intern_equality_against_str() {
        let shared = "compare".intern().unwrap();
        assert_eq!(shared, "compare");
        assert_ne!(shared, "different");
    }

    #[test]
    fn intern_owned_method() {
        let a = <str as InternExt>::intern_owned("owned method".to_string()).unwrap();
        let b = "owned method".intern().unwrap();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn intern_debug() {
        let s = "debug me".intern().unwrap();
        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("Shared"));
    }

    #[test]
    fn intern_concurrent_access() {
        use lang_std::thread;
        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    let key = format!("key{}", i);
                    let interned = key.as_str().intern().unwrap();
                    assert_eq!(&*interned, format!("key{}", i));
                    interned
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn intern_promote_owned_to_shared() {
        let owned = Intern::<str>::from_owned("promote_me".to_string());
        assert!(owned.is_owned());
        let promoted = owned.promote();
        assert!(promoted.is_shared());
        assert_eq!(&*promoted, "promote_me");
    }

    #[test]
    fn intern_promote_already_shared() {
        let shared = "already_shared".intern().unwrap();
        let promoted = shared.promote();
        assert!(promoted.is_shared());
    }

    #[test]
    fn intern_osstr_basic() {
        let os = OsStr::new("os_test");
        let a = os.intern().unwrap();
        assert!(a.is_shared());
        assert_eq!(&*a, os);
    }

    #[test]
    fn intern_osstr_deduplication() {
        let os1 = OsStr::new("dedup_os");
        let os2 = OsStr::new("dedup_os");
        let a = os1.intern().unwrap();
        let b = os2.intern().unwrap();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn intern_cstr_basic() {
        let cs = c"ctest";
        let a = cs.intern().unwrap();
        assert!(a.is_shared());
        assert_eq!(&*a, cs);
    }

    #[test]
    fn intern_cstr_deduplication() {
        let cs1 = c"dedup_cs";
        let cs2 = c"dedup_cs";
        let a = cs1.intern().unwrap();
        let b = cs2.intern().unwrap();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn intern_path_basic() {
        let p = Path::new("/tmp/test");
        let a = p.intern().unwrap();
        assert!(a.is_shared());
        assert_eq!(&*a, p);
    }

    #[test]
    fn intern_path_deduplication() {
        let p1 = Path::new("/usr/local/bin");
        let p2 = Path::new("/usr/local/bin");
        let a = p1.intern().unwrap();
        let b = p2.intern().unwrap();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn intern_path_promote() {
        let owned = Intern::<Path>::from_owned(PathBuf::from("/var/log"));
        assert!(owned.is_owned());
        let promoted = owned.promote();
        assert!(promoted.is_shared());
        assert_eq!(&*promoted, Path::new("/var/log"));
    }

    #[test]
    fn intern_clone_works() {
        let a = "clone_test".intern().unwrap();
        let b = a.clone();
        assert!(a.ptr_eq(&b));
    }
}
