use std::sync::atomic::{AtomicBool, Ordering};
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
pub fn check() -> crate::error::Result<()> {
    if INTERRUPTED.load(Ordering::Relaxed) {
        Err(crate::error::Error {
            kind: "KeyboardInterrupt",
            code: None,
            message: "interrupted".into(),
        })
    } else {
        Ok(())
    }
}

#[cfg(windows)]
pub fn install() {
    unsafe extern "system" fn handler(code: u32) -> i32 {
        if code <= 1 {
            INTERRUPTED.store(true, Ordering::Relaxed);
            1
        } else {
            0
        }
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }
    unsafe {
        SetConsoleCtrlHandler(Some(handler), 1);
    }
}

#[cfg(not(windows))]
pub fn install() {
    extern "C" fn handler(_: i32) {
        INTERRUPTED.store(true, Ordering::Relaxed);
    }
    unsafe extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    unsafe {
        signal(2, handler);
    }
}
