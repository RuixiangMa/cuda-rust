/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Device, DriverError, IntoResult};
use std::ffi::{c_int, c_void};

fn has_gpu() -> bool {
    Device::device_count().map(|n| n > 0).unwrap_or(false)
}

#[test]
fn event_record_rejects_different_real_context() {
    if !has_gpu() {
        return;
    }

    let primary = Device::new(0).expect("device");
    let event = primary.new_event().expect("event");
    let device = primary.cu_device();

    let mut secondary = std::ptr::null_mut();
    unsafe {
        cuda_bindings::cuCtxCreate_v4(&mut secondary, std::ptr::null_mut(), 0, device)
            .result()
            .expect("create secondary context");
    }

    let secondary_device =
        unsafe { Device::borrow_raw(secondary.cast::<c_void>(), device as c_int, 0) };
    let stream = secondary_device.new_stream().expect("secondary stream");
    let error = event
        .record(&stream)
        .expect_err("record across contexts must be rejected");

    assert_eq!(
        error,
        DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE)
    );

    drop(stream);
    unsafe {
        cuda_bindings::cuCtxDestroy_v2(secondary)
            .result()
            .expect("destroy secondary context");
    }
    primary.bind_to_thread().expect("restore primary context");
}
