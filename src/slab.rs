use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::Ordering;
use core::{marker::PhantomData, ptr::NonNull, result::Result, sync::atomic::AtomicU8};

/// Atomically test and set a bit in an array.
fn atomic_bts(arr: &[AtomicU8], idx: usize) -> Option<bool> {
    let el = arr.get(idx / 8)?;
    let bit = (idx % 8) as u8;

    let val = el.fetch_or(1 << bit, Ordering::Relaxed);
    Some((val & (1 << bit)) != 0)
}

/// Atomically test and clear a bit in an array.
fn atomic_btc(arr: &[AtomicU8], idx: usize) -> Option<bool> {
    let el = arr.get(idx / 8)?;
    let bit = (idx % 8) as u8;

    let val = el.fetch_and(!(1 << bit), Ordering::Relaxed);
    Some((val & (1 << bit)) != 0)
}

/// Perform a division, preferring to round up.
fn div_ceil(num: usize, dem: usize) -> usize {
    (num + dem) / dem
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabError {
    /// The base address of the heap is not aligned properly.
    BadBaseAlignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabAllocError {
    /// The heap was observed to be exhausted at the time of the allocation.
    HeapExhausted,
}

unsafe impl<T> Send for SlabAllocator<T> {}
unsafe impl<T> Sync for SlabAllocator<T> {}

/// A basic fixed-size slab allocator covering a range of memory.
#[derive(Debug)]
pub struct SlabAllocator<T: Sized> {
    _data: PhantomData<T>,

    /// The total number of elements in this slab allocator.
    num_elems: usize,
    /// The bitmap slice.
    bitmap: &'static mut [AtomicU8],
    /// A pointer to the data block, which contains `num_elems` elements.
    data: *mut MaybeUninit<T>,
}

impl<T: Sized> SlabAllocator<T> {
    /// Create a new instance of this allocator.
    /// If the input parameters are invalid, this will return a [SlabError].
    pub fn new(mem: *mut MaybeUninit<u8>, size: usize) -> Result<Self, SlabError> {
        // Verify the base address is aligned properly.
        if (mem as usize) & (align_of::<T>() - 1) != 0 {
            return Err(SlabError::BadBaseAlignment);
        }

        let element_size = core::mem::size_of::<T>();

        // Calculate the size of the data segment, subtracting out the ideal
        // bitmap size.
        let data_size = size - div_ceil(size / element_size, u8::BITS as usize);

        // Partition off the data first.
        // FIXME: Does this ensure the alignment of elements?
        let data = mem as *mut MaybeUninit<T>;

        // Calculate the actual number of elements that can be stored in the data segment.
        let num_elems = data_size / element_size;
        let bitmap_size = div_ceil(num_elems, u8::BITS as usize);

        // Slice off the bitmap, taking care to initialize it.
        let bitmap = {
            let bitmap = unsafe {
                core::slice::from_raw_parts_mut(
                    mem.add(size - bitmap_size) as *mut MaybeUninit<AtomicU8>,
                    bitmap_size,
                )
            };

            for b in bitmap.iter_mut() {
                b.write(AtomicU8::new(0));
            }

            // The memory is now initialized.
            unsafe {
                core::slice::from_raw_parts_mut(bitmap.as_mut_ptr() as *mut AtomicU8, bitmap.len())
            }
        };

        Ok(Self {
            _data: PhantomData,

            num_elems,
            bitmap,
            data,
        })
    }

    /// Attempts to allocate a slot in our bitmap.
    fn allocate_slot(&self) -> Option<usize> {
        for i in 0..self.num_elems {
            if let Some(false) = atomic_bts(self.bitmap, i) {
                return Some(i);
            }
        }

        None
    }

    fn deallocate_slot(&self, slot: usize) {
        atomic_btc(self.bitmap, slot);
    }

    fn allocate_raw(&self) -> Result<&mut MaybeUninit<T>, SlabAllocError> {
        let slot = self.allocate_slot().ok_or(SlabAllocError::HeapExhausted)?;

        let data = unsafe { &mut *self.data.add(slot) };

        // Since we have an exclusive slot, we can safely pass out a mutable pointer.
        Ok(&mut *data)
    }

    unsafe fn deallocate_raw(&self, elm: *mut T) {
        // Calculate the slot by finding the offset from our data buffer.
        let slot = elm.offset_from(self.data as *const T);
        if slot < 0 || (slot as usize) >= self.num_elems {
            panic!("Deallocate given an element that does not belong to us!");
        }

        self.deallocate_slot(slot as usize);
    }

    pub fn allocate<'a>(&'a self, item: T) -> Result<SlabAlloc<'a, T>, SlabAllocError> {
        let e = unsafe {
            let e = self.allocate_raw()?;
            e.as_mut_ptr().write(item);
            e.assume_init_mut()
        };

        Ok(SlabAlloc(self, unsafe { NonNull::new_unchecked(e) }))
    }

    /// Constructs a `SlabAlloc` from a raw pointer.
    ///
    /// This function is analogous to [`Box::from_raw`].
    ///
    /// # Safety
    ///
    /// The raw pointer must be non-null, properly aligned, and have been previously
    /// returned by a call to [`SlabAlloc::leak`] on an allocation from this allocator.
    pub unsafe fn from_raw<'a>(&'a self, ptr: *mut T) -> SlabAlloc<'a, T> {
        SlabAlloc(self, NonNull::new_unchecked(ptr))
    }
}

/// A container structure for an allocation made in a [SlabAllocator].
/// This is similar to a [core::alloc::Box].
#[derive(Debug)]
pub struct SlabAlloc<'a, T>(&'a SlabAllocator<T>, NonNull<T>);

impl<'a, T> SlabAlloc<'a, T> {
    /// Leaks the allocation, returning a mutable pointer to the contained value.
    /// The returned pointer will be non-null and properly aligned.
    ///
    /// This function is analogous to [`Box::leak`].
    pub fn leak(self) -> *mut T {
        let ptr = self.1.as_ptr();
        core::mem::forget(self);
        ptr
    }
}

impl<T> Drop for SlabAlloc<'_, T> {
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(self.1.as_ptr());
            self.0.deallocate_raw(self.1.as_ptr())
        };
    }
}

impl<T> Deref for SlabAlloc<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.1.as_ref() }
    }
}

impl<T> DerefMut for SlabAlloc<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.1.as_mut() }
    }
}

#[cfg(test)]
mod test {
    // Allow use of std in tests.
    extern crate std;

    use core::alloc::Layout;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn test_atomic_bt() {
        use super::*;

        let arr = vec![AtomicU8::new(0), AtomicU8::new(0)];

        // Twiddle a single bit.
        assert_eq!(Some(false), atomic_bts(&arr, 0));
        assert_eq!(Some(true), atomic_bts(&arr, 0));
        assert_eq!(Some(true), atomic_btc(&arr, 0));
        assert_eq!(Some(false), atomic_bts(&arr, 0));
        assert_eq!(Some(true), atomic_btc(&arr, 0));

        // Twiddle two bits, ensuring the two do not interfere with one another.
        assert_eq!(Some(false), atomic_bts(&arr, 1));
        assert_eq!(Some(false), atomic_bts(&arr, 2));
        assert_eq!(Some(true), atomic_bts(&arr, 2));
        assert_eq!(Some(true), atomic_btc(&arr, 2));
        assert_eq!(Some(true), atomic_bts(&arr, 1));
        assert_eq!(Some(true), atomic_btc(&arr, 1));

        // Ditto, in two separate bytes.
        assert_eq!(Some(false), atomic_bts(&arr, 0));
        assert_eq!(Some(false), atomic_bts(&arr, 8));
        assert_eq!(Some(true), atomic_bts(&arr, 8));
        assert_eq!(Some(true), atomic_btc(&arr, 8));
        assert_eq!(Some(true), atomic_bts(&arr, 0));
        assert_eq!(Some(true), atomic_btc(&arr, 0));

        // Test OOB access.
        assert_eq!(None, atomic_bts(&arr, 20));
    }

    #[test]
    fn test_basic() {
        use super::*;

        #[derive(Debug)]
        struct Element {
            data: usize,
        }

        let layout = Layout::from_size_align(16384, 128).unwrap();
        let mem = unsafe { std::alloc::alloc(layout) };

        let alloc: SlabAllocator<Element> =
            SlabAllocator::new(mem as *mut MaybeUninit<u8>, layout.size()).unwrap();

        let mut allocs = Vec::new();

        // N.B: There should be 2016 slots available.
        for i in 0..32 {
            allocs.push(alloc.allocate(Element { data: i as usize }).unwrap());
        }

        drop(allocs);
        drop(alloc);
        unsafe { std::alloc::dealloc(mem, layout) };
    }

    /// Test a small slab allocator and ensure it properly reports heap exhaustion.
    #[test]
    fn test_small() {
        use super::*;

        #[derive(Debug)]
        struct Element {
            data: usize,
        }

        let layout = Layout::from_size_align(32, 128).unwrap();
        let mem = unsafe { std::alloc::alloc(layout) };

        let alloc: SlabAllocator<Element> =
            SlabAllocator::new(mem as *mut MaybeUninit<u8>, layout.size()).unwrap();

        let mut allocs = Vec::new();

        // N.B: There are only 3 slots available.
        allocs.push(alloc.allocate(Element { data: 0 as usize }).unwrap());
        allocs.push(alloc.allocate(Element { data: 1 as usize }).unwrap());
        allocs.push(alloc.allocate(Element { data: 2 as usize }).unwrap());
        assert_eq!(
            SlabAllocError::HeapExhausted,
            alloc.allocate(Element { data: 3 as usize }).unwrap_err()
        );

        assert_eq!(allocs[0].data, 0);
        assert_eq!(allocs[1].data, 1);
        assert_eq!(allocs[2].data, 2);

        drop(allocs);
        drop(alloc);
        unsafe { std::alloc::dealloc(mem, layout) };
    }

    #[test]
    fn test_leak() {
        use super::*;

        #[derive(Debug, PartialEq, Eq)]
        struct Element {
            data: usize,
        }

        let layout = Layout::from_size_align(1024, 8).unwrap();
        let mem = unsafe { std::alloc::alloc(layout) };

        let alloc: SlabAllocator<Element> =
            SlabAllocator::new(mem as *mut MaybeUninit<u8>, layout.size()).unwrap();

        let num_elems = alloc.num_elems;
        assert!(num_elems > 0);

        // Allocate all elements
        let mut allocs = Vec::new();
        for i in 0..num_elems {
            allocs.push(alloc.allocate(Element { data: i }).unwrap());
        }

        // Ensure the next allocation fails.
        assert_eq!(
            SlabAllocError::HeapExhausted,
            alloc.allocate(Element { data: 0 }).unwrap_err()
        );

        // Leak half of the allocations, and drop the other half.
        let num_to_leak = num_elems / 2;
        let num_to_drop = num_elems - num_to_leak;

        let mut leaked = Vec::new();
        for a in allocs.drain(0..num_to_leak) {
            leaked.push(a.leak());
        }

        // The remaining elements are dropped here.
        drop(allocs);

        // We should be able to re-allocate the dropped elements.
        let mut new_allocs = Vec::new();
        for i in 0..num_to_drop {
            new_allocs.push(alloc.allocate(Element { data: i }).unwrap());
        }

        // Ensure the next allocation fails.
        assert_eq!(
            SlabAllocError::HeapExhausted,
            alloc.allocate(Element { data: 0 }).unwrap_err()
        );

        // Now, reconstruct and drop the leaked pointers.
        for (i, p) in leaked.into_iter().enumerate() {
            let b = unsafe { alloc.from_raw(p) };
            assert_eq!(b.data, i);
            drop(b);
        }

        // We should be able to allocate "num_to_leak" more elements.
        let mut reclaimed_allocs = Vec::new();
        for i in 0..num_to_leak {
            reclaimed_allocs.push(alloc.allocate(Element { data: i }).unwrap());
        }

        // Ensure the next allocation fails.
        assert_eq!(
            SlabAllocError::HeapExhausted,
            alloc.allocate(Element { data: 0 }).unwrap_err()
        );

        // Drop all allocations
        drop(new_allocs);
        drop(reclaimed_allocs);

        // We should be able to allocate everything again.
        let mut final_allocs = Vec::new();
        for i in 0..num_elems {
            final_allocs.push(alloc.allocate(Element { data: i }).unwrap());
        }

        // Ensure the next allocation fails.
        assert_eq!(
            SlabAllocError::HeapExhausted,
            alloc.allocate(Element { data: 0 }).unwrap_err()
        );

        drop(final_allocs);
        drop(alloc);
        unsafe { std::alloc::dealloc(mem, layout) };
    }
}
