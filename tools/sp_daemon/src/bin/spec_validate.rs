/// spec_validate scaffold — compile-time check that sp_session_rewind is bound.
/// Replaced with full implementation in Task 5.

mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/sp_bindings.rs"));
}

fn main() {
    let _: fn(*mut ffi::sp_session, usize) = |ptr, n| {
        let st = unsafe { ffi::sp_session_rewind(ptr, n) };
        assert_eq!(st, ffi::sp_status_SP_OK);
    };
    println!("spec_validate compile OK");
}
