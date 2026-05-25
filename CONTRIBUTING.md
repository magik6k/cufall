# Contributing

Contributions are welcome. Keep changes small, focused, and easy to verify.

Before sending a change:

1. Build with `make`.
2. Run `./cufall --help` and check that CLI help still renders cleanly.
3. If you changed CUPTI setup or metric handling, test on a machine with NVIDIA performance-counter access.

Notes for metric changes:

- CUPTI/PerfWorks metric names vary by GPU architecture and CUPTI release.
- Prefer examples that fail cleanly at startup when unsupported.
- Keep default sampling overhead low.
