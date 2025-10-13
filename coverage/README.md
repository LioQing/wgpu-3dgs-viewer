# Coverage

This directory contains code coverage reports.

The coverage reports are generated using the [run.ps1](./run.ps1) script. To run the script, you need to install [cargo-llvm-cov](https://crates.io/crates/cargo-llvm-cov) and [cargo-nextest](https://crates.io/crates/cargo-nextest).

> [!NOTE]
>
> Because the crate requires GPU support, the coverage report is not run in CI.