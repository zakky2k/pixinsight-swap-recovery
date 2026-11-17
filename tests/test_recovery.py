"""Regression tests for the established PixInsight parallel-swap layout."""
import contextlib
import io
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pi_swap_to_xisf as recoverer


FORMAT_SIGNATURES = {
    "Float32": bytes.fromhex("0101002001000000"),
    "UInt16": bytes.fromhex("0100001001000000"),
}


def make_swap(folder, name, sample_format, channels, width=17, height=11,
              sample_counts=(23, 29, 135)):
    """Build an artificial swap group with independently generated pixel planes."""
    sample_bytes = 4 if sample_format == "Float32" else 2
    plane_bytes = width * height * sample_bytes
    # Per-segment channel blocks must have whole samples; not whole rows.
    assert sum(sample_counts) == width * height
    lengths = [count * sample_bytes for count in sample_counts]
    planes = [bytes((i * 31 + channel * 19) % 256
                    for i in range(plane_bytes)) for channel in range(channels)]
    primary = folder / f"~PI~TOKEN~{name}.swp"
    consumed = 0
    for idx, length in enumerate(lengths):
        path = primary if idx == 0 else primary.with_name(
            f"{primary.stem}-part{idx:03d}.swp")
        with path.open("wb") as f:
            if idx == 0:
                header = bytearray(128)
                header[:8] = b"egamiwar"
                header[12:20] = FORMAT_SIGNATURES[sample_format]
                struct.pack_into("<III", header, 0x1c, width, height, channels)
                struct.pack_into("<Q", header, 0x68, len(lengths))
                f.write(header)
            for channel_plane in planes:
                f.write(channel_plane[consumed:consumed + length])
        consumed += length
    return primary, b"".join(planes)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = pathlib.Path(self.temp.name)

    def quiet(self, callable_, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return callable_(*args, **kwargs)

    def test_round_trip_all_four_formats(self):
        for sample_format in ("Float32", "UInt16"):
            for channels in (1, 3):
                with self.subTest(sample_format=sample_format, channels=channels):
                    primary, expected = make_swap(
                        self.folder, f"{sample_format}_{channels}",
                        sample_format, channels)
                    output = self.folder / f"{sample_format}_{channels}.xisf"
                    self.quiet(recoverer.recover, primary, output)
                    with output.open("rb") as f:
                        f.seek(recoverer.ATTACHMENT_OFFSET)
                        self.assertEqual(f.read(), expected)
                    self.quiet(recoverer.validate, output)
                    self.quiet(recoverer.repair_header, output)
                    with output.open("rb") as f:
                        f.seek(recoverer.ATTACHMENT_OFFSET)
                        self.assertEqual(f.read(), expected)

    def test_missing_segment_is_rejected(self):
        primary, _ = make_swap(self.folder, "MissingPart", "Float32", 3)
        missing = self.folder / f"{primary.stem}-part001.swp"
        missing.unlink()
        output = self.folder / "missing.xisf"
        with self.assertRaises(FileNotFoundError):
            self.quiet(recoverer.recover, primary, output)
        self.assertFalse(output.exists())

    def test_existing_destination_is_not_overwritten(self):
        primary, _ = make_swap(self.folder, "Exists", "UInt16", 3)
        output = self.folder / "existing.xisf"
        output.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.quiet(recoverer.recover, primary, output)
        self.assertEqual(output.read_bytes(), b"original")

    def test_bad_signature_is_rejected(self):
        primary, _ = make_swap(self.folder, "Bad", "Float32", 1)
        with primary.open("r+b") as f:
            f.seek(12)
            f.write(b"BADFMT!!")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.quiet(recoverer.inspect_swap, primary)

    def test_batch_dry_run_creates_no_output(self):
        swap_dir = self.folder / "swaps"
        swap_dir.mkdir()
        output_dir = self.folder / "recovered"
        make_swap(swap_dir, "DryRun", "UInt16", 3)
        self.assertEqual(self.quiet(recoverer.recover_all,
                                    swap_dir, output_dir, True), 0)
        self.assertFalse(output_dir.exists())

    def test_batch_recovers_and_preserves_prior_output(self):
        swap_dir = self.folder / "swaps"
        swap_dir.mkdir()
        output_dir = self.folder / "recovered"
        _, gray = make_swap(swap_dir, "SameName", "Float32", 1)
        # Second swap group with same user-visible name and unique token.
        for segment in sorted(swap_dir.glob("~PI~TOKEN~SameName*.swp")):
            segment.rename(segment.with_name(segment.name.replace("TOKEN", "TOK2")))
        _, rgb = make_swap(swap_dir, "SameName", "UInt16", 3)
        self.assertEqual(self.quiet(recoverer.recover_all,
                                    swap_dir, output_dir, False), 0)
        self.assertEqual(len(list(output_dir.glob("*.xisf"))), 2)
        self.assertEqual(self.quiet(recoverer.recover_all,
                                    swap_dir, output_dir, False), 0)
        payloads = {
            path.name: path.read_bytes()[recoverer.ATTACHMENT_OFFSET:]
            for path in output_dir.glob("*.xisf")
        }
        self.assertEqual(payloads, {
            "Recovered_SameName.xisf": gray,
            "Recovered_SameName_TOK2.xisf": gray,
            "Recovered_SameName_TOKEN.xisf": rgb,
            "Recovered_SameName_TOKEN_2.xisf": rgb,
        })

    def test_single_sample_strips_and_unaligned_copy_chunks(self):
        for sample_format in ("Float32", "UInt16"):
            with self.subTest(sample_format=sample_format):
                primary, expected = make_swap(
                    self.folder, sample_format, sample_format, 3,
                    width=7, height=2, sample_counts=(1, 8, 1, 4))
                output = self.folder / f"{sample_format}.xisf"
                with patch.object(recoverer, "CHUNK_SIZE", 3):
                    self.quiet(recoverer.recover, primary, output)
                self.assertEqual(output.read_bytes()[recoverer.ATTACHMENT_OFFSET:], expected)

    def test_rejects_invalid_rgb_boundaries_even_with_matching_total(self):
        primary, _ = make_swap(self.folder, "Split", "UInt16", 3)
        first = primary.read_bytes()
        primary.write_bytes(first[:-1])
        continuation = primary.with_name(f"{primary.stem}-part001.swp")
        continuation.write_bytes(first[-1:] + continuation.read_bytes())
        output = self.folder / "split.xisf"
        with self.assertRaisesRegex(ValueError, "channel"):
            self.quiet(recoverer.recover, primary, output)
        with self.assertRaisesRegex(ValueError, "channel"):
            self.quiet(recoverer.inspect_swap, primary)
        self.assertFalse(output.exists())

    def test_batch_skips_missing_segments_and_recovers_other_groups(self):
        primary, _ = make_swap(self.folder, "Missing", "UInt16", 3)
        primary.with_name(f"{primary.stem}-part002.swp").unlink()
        make_swap(self.folder, "Complete", "Float32", 1)
        out = self.folder / "output"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(recoverer.recover_all(self.folder, out), 0)
        self.assertIn("Missing continuation segment", stream.getvalue())
        self.assertIn("1 recovered, 1 skipped, 0 failed", stream.getvalue())
        self.assertEqual([p.name for p in out.iterdir()], ["Recovered_Complete.xisf"])

    def test_rejects_malformed_swap_headers(self):
        primary, _ = make_swap(self.folder, "BadHeader", "Float32", 1)
        original = primary.read_bytes()
        output = self.folder / "invalid.xisf"
        for label, invalid in (
            ("truncated", original[:127]),
            ("magic", b"badmagic" + original[8:]),
            ("zero width", original[:28] + bytes(4) + original[32:]),
            ("channel count", original[:36] + struct.pack("<I", 2) + original[40:]),
            ("segment count", original[:104] + struct.pack("<Q", 257) + original[112:]),
        ):
            with self.subTest(label=label):
                primary.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    self.quiet(recoverer.inspect_swap, primary)
                with self.assertRaises(ValueError):
                    self.quiet(recoverer.recover, primary, output)
                self.assertFalse(output.exists())

    def test_invalid_xisf_is_rejected_before_repair_writes(self):
        primary, _ = make_swap(self.folder, "Repair", "Float32", 1)
        output = self.folder / "repair.xisf"
        self.quiet(recoverer.recover, primary, output)
        original = output.read_bytes()
        for valid, invalid in (
            (b'byteOrder="little"', b'byteOrder="big"   '),
            (b'bounds="0:1"', b'bounds="1:0"'),
            (b'colorSpace="Gray"', b'colorSpace="RGB" '),
        ):
            with self.subTest(invalid=invalid):
                broken = original.replace(valid, invalid, 1)
                output.write_bytes(broken)
                with self.assertRaises(ValueError):
                    self.quiet(recoverer.validate, output)
                with self.assertRaises(ValueError):
                    self.quiet(recoverer.repair_header, output)
                self.assertEqual(output.read_bytes(), broken)

    def test_failed_exclusive_open_preserves_competing_output(self):
        primary, _ = make_swap(self.folder, "Race", "UInt16", 1)
        output = self.folder / "competing.xisf"
        real_open = pathlib.Path.open

        def create_competitor(path, mode="r", *args, **kwargs):
            if path == output and mode == "xb":
                with real_open(path, "wb") as competing:
                    competing.write(b"other process")
            return real_open(path, mode, *args, **kwargs)

        with patch.object(pathlib.Path, "open", create_competitor):
            with self.assertRaises(FileExistsError):
                self.quiet(recoverer.recover, primary, output)
        self.assertEqual(output.read_bytes(), b"other process")

    def test_cli_version_and_missing_pair(self):
        with self.assertRaises(SystemExit) as version:
            self.quiet(recoverer.main, ["--version"])
        self.assertEqual(version.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                self.quiet(recoverer.main, ["some.swp"])
        self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
