# Changelog

## 1.0.0 — 2026-11-17

- Preserve the successful Float32 / UInt16, Gray / RGB parallel-swap recovery paths.
- Maintain single-image, batch, dry-run, XISF validation and header repair commands.
- Add a versioned command-line interface, better CLI validation and concise errors.
- Add byte-for-byte synthetic regression coverage for uneven channel blocks, duplicate names, missing segments and malformed headers.
- Protect existing output files during concurrent recovery and validate candidate legacy headers before writing repairs.
- Document Windows PowerShell usage, compatibility boundaries and data-loss precautions.

The binary swap format is undocumented; published compatibility is limited to the layouts described in the README.
