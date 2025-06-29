#![doc = include_str!("../README.md")]
#![no_std]
#![feature(maybe_uninit_slice)]

mod slab;
pub use slab::{SlabAlloc, SlabAllocError, SlabAllocator};
