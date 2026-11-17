#!/usr/bin/env python3
"""
PixInsight Parallel Swap to XISF Recovery
Recovers uncompressed Float32 and UInt16 grayscale/RGB images from
PixInsight parallel swap files into XISF 1.0 format.
Supports:
    - Multiple parallel swap segments
    - Direct conversion to XISF
    - Repair of previously generated XISF headers
    - Structural validation of XISF files
    - Batch recovery of all supported images in a swap directory
    - Float32 RGB image recovery (3-channel planar data)
    - UInt16 grayscale/RGB image recovery with original precision
    - Reorder per-segment RGB channel blocks into full XISF planes
    - Dry-run preview and per-image failure reporting
    - No overwrite of existing XISF outputs
No third-party Python dependencies required.
IMPORTANT:
    Always preserve your original swap files.
    Back up any XISF before using --repair-header.
    Batch mode should be run against a backup of an inactive swap directory.
The swap format is undocumented. RGB output is assembled by joining the
corresponding channel blocks from each segment into complete image planes.
"""
import argparse
import math
import sys
from pathlib import Path
import re
import struct
import os
import tempfile
import xml.etree.ElementTree as ET

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SWAP_HEADER_SIZE = 128
ATTACHMENT_OFFSET = 4096
CHUNK_SIZE = 4 * 1024 * 1024
XISF_SIGNATURE = b"XISF0100"
XISF_NAMESPACE = "http://www.pixinsight.com/xisf"
VERSION = "1.0.0"

# Known swap sample format identifiers and XISF sample sizes.
# Keep the established Float32 signature unchanged.
SWAP_PIXEL_FORMATS = {
    bytes.fromhex("0101002001000000"): ("Float32", 4),
    bytes.fromhex("0100001001000000"): ("UInt16", 2),
}


def swap_sample_format(header):
    """Return (XISF sample format, bytes per sample) or reject unknown data."""
    signature = header[0x0c:0x14]
    if signature not in SWAP_PIXEL_FORMATS:
        raise ValueError(
            f"Unsupported PixInsight swap pixel format: {signature.hex(' ')}"
        )
    return SWAP_PIXEL_FORMATS[signature]


# --------------------------------------------------
# XISF XML header
# --------------------------------------------------

def create_xml(width, height, data_size,
               image_id="RecoveredImage", channels=1, sample_format="Float32"):
    if channels not in (1, 3):
        raise ValueError("Only grayscale and RGB images are supported")
    if sample_format not in ("Float32", "UInt16"):
        raise ValueError("Unsupported XISF sample format")
    color_space = "Gray" if channels == 1 else "RGB"
    # Floating point pixels need explicit normalized bounds in PixInsight.
    # Integer XISF images use their native range and omit bounds.
    bounds_attribute = 'bounds="0:1" ' if sample_format == "Float32" else ''
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xisf xmlns="http://www.pixinsight.com/xisf" '
        'version="1.0">\n'
        f'  <Image id="{image_id}" '
        f'geometry="{width}:{height}:{channels}" '
        f'sampleFormat="{sample_format}" '
        f'{bounds_attribute}'
        f'colorSpace="{color_space}" '
        'pixelStorage="Planar" '
        'byteOrder="little" '
        f'location="attachment:{ATTACHMENT_OFFSET}:'
        f'{data_size}"/>\n'
        '  <Metadata/>\n'
        '</xisf>\n'
    )
    return xml.encode("utf-8")

# --------------------------------------------------
# XISF validation
# --------------------------------------------------

def validate(path):
    path = Path(path)
    with path.open("rb") as f:
        preamble = f.read(16)
        if len(preamble) != 16:
            raise ValueError("Incomplete XISF preamble")
        if preamble[:8] != XISF_SIGNATURE:
            raise ValueError("Invalid XISF signature")
        header_length, reserved = struct.unpack_from(
            "<II", preamble, 8
        )
        if reserved != 0:
            raise ValueError("Invalid reserved header field")
        if not 0 < header_length <= ATTACHMENT_OFFSET - 16:
            raise ValueError("Invalid XML header length")
        xml = f.read(header_length)
        file_size = os.fstat(f.fileno()).st_size
    width, height, sample_format, color_space, data_size = validate_header(xml, file_size)
    print(
        f"Validated: {width}x{height}, "
        f"{sample_format} {color_space}, {data_size:,} image bytes"
    )


def validate_header(xml, file_size):
    # PixInsight can reject namespace-prefixed roots
    # such as <ns0:xisf>. Require an unprefixed root.
    if not re.search(rb"<xisf(?:\s|>)", xml):
        raise ValueError(
            "XISF XML must contain an unprefixed "
            "<xisf> root element"
        )
    root = ET.fromstring(xml)
    expected_root = f"{{{XISF_NAMESPACE}}}xisf"
    if root.tag != expected_root:
        raise ValueError("Invalid XISF XML namespace")
    if root.get("version") != "1.0":
        raise ValueError("Invalid XISF version")
    image_tag = f"{{{XISF_NAMESPACE}}}Image"
    metadata_tag = f"{{{XISF_NAMESPACE}}}Metadata"
    images = root.findall(image_tag)
    if len(images) != 1:
        raise ValueError(
            "Expected exactly one image element"
        )
    if root.find(metadata_tag) is None:
        raise ValueError(
            "Missing XISF Metadata element"
        )
    image = images[0]
    required = [
        "geometry",
        "sampleFormat",
        "colorSpace",
        "location"
    ]
    for attribute in required:
        if attribute not in image.attrib:
            raise ValueError(
                f"Missing mandatory attribute: {attribute}"
            )
    sample_format = image.get("sampleFormat")
    if sample_format not in ("Float32", "UInt16"):
        raise ValueError("Unsupported pixel format")
    bytes_per_sample = 4 if sample_format == "Float32" else 2
    if image.get("colorSpace") not in ("Gray", "RGB"):
        raise ValueError("Unsupported colour space")
    dimensions = tuple(
        int(x) for x in image.get("geometry").split(":")
    )
    if len(dimensions) != 3:
        raise ValueError("Invalid image geometry")
    width, height, channels = dimensions
    if width <= 0 or height <= 0 or channels not in (1, 3):
        raise ValueError("Unsupported image dimensions")
    if image.get("colorSpace") != ("Gray" if channels == 1 else "RGB"):
        raise ValueError("Channel count and colour space do not match")
    if image.get("pixelStorage", "Planar") != "Planar":
        raise ValueError("Unsupported pixel storage")
    if image.get("byteOrder", "little") != "little":
        raise ValueError("Unsupported byte order")
    if "compression" in image.attrib:
        raise ValueError("Unsupported compressed attachment")
    if sample_format == "Float32":
        if "bounds" not in image.attrib:
            raise ValueError("Missing mandatory Float32 bounds")
        bounds = tuple(
            float(x) for x in image.get("bounds").split(":")
        )
        if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("Invalid floating-point bounds")
    location = image.get("location").split(":")
    if len(location) != 3:
        raise ValueError("Invalid attachment location")
    if location[0] != "attachment":
        raise ValueError("Unsupported attachment type")
    offset = int(location[1])
    data_size = int(location[2])
    expected_size = width * height * channels * bytes_per_sample
    if data_size != expected_size:
        raise ValueError(
            "Image attachment size does not match dimensions"
        )
    if offset < 16 + len(xml):
        raise ValueError(
            "Image attachment overlaps XML header"
        )
    if file_size != offset + data_size:
        raise ValueError("XISF file size does not match attachment")
    return width, height, sample_format, image.get("colorSpace"), data_size


# --------------------------------------------------
# Repair existing XISF header
# --------------------------------------------------

def repair_header(path):
    path = Path(path)
    with path.open("r+b") as f:
        preamble = f.read(16)
        if len(preamble) != 16:
            raise ValueError("Incomplete XISF preamble")
        if preamble[:8] != XISF_SIGNATURE:
            raise ValueError("Not an XISF file")
        header_length, reserved = struct.unpack_from(
            "<II", preamble, 8
        )
        if reserved != 0:
            raise ValueError("Invalid reserved field")
        if not 0 < header_length <= ATTACHMENT_OFFSET - 16:
            raise ValueError("Invalid header length")
        xml = f.read(header_length)
        root = ET.fromstring(xml)
        expected_root = f"{{{XISF_NAMESPACE}}}xisf"
        if root.tag != expected_root:
            raise ValueError("Unexpected XML root")
        if root.get("version") != "1.0":
            raise ValueError("Unsupported XISF version")
        image_tag = f"{{{XISF_NAMESPACE}}}Image"
        metadata_tag = f"{{{XISF_NAMESPACE}}}Metadata"
        images = root.findall(image_tag)
        if len(images) != 1:
            raise ValueError("Expected exactly one image")
        image = images[0]
        sample_format = image.get("sampleFormat")
        if sample_format not in ("Float32", "UInt16"):
            raise ValueError("Unsupported sample format")
        if sample_format == "Float32" and "bounds" not in image.attrib:
            image.set("bounds", "0:1")
        if root.find(metadata_tag) is None:
            ET.SubElement(root, metadata_tag)
        # CRITICAL FIX:
        #
        # Without this registration ElementTree emits:
        #
        # <ns0:xisf xmlns:ns0="...">
        #
        # This can be rejected by PixInsight.
        #
        # Register the default namespace so the root is:
        #
        # <xisf xmlns="..." version="1.0">
        ET.register_namespace("", XISF_NAMESPACE)
        repaired_xml = ET.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True
        )
        if not re.search(
            rb"<xisf(?:\s|>)",
            repaired_xml
        ):
            raise ValueError(
                "XML serializer produced invalid XISF root"
            )
        if len(repaired_xml) + 16 > ATTACHMENT_OFFSET:
            raise ValueError(
                "Repaired header exceeds reserved space"
            )
        location = image.get("location", "").split(":")
        if len(location) != 3:
            raise ValueError("Unexpected attachment location")
        if location[0] != "attachment":
            raise ValueError("Unsupported attachment type")
        offset = int(location[1])
        if offset != ATTACHMENT_OFFSET:
            raise ValueError(
                "Cannot repair unexpected attachment offset"
            )
        validate_header(repaired_xml, os.fstat(f.fileno()).st_size)
        # Update only the XISF header.
        # The image attachment remains unchanged.
        f.seek(8)
        f.write(
            struct.pack("<I", len(repaired_xml))
        )
        f.seek(16)
        f.write(repaired_xml)
        padding = (
            ATTACHMENT_OFFSET - 16 - len(repaired_xml)
        )
        f.write(b"\0" * padding)
    validate(path)
    print("Header repair completed successfully.")

# --------------------------------------------------
# Recover PixInsight parallel swap
# --------------------------------------------------

def recover(primary, output):
    primary = Path(primary).resolve()
    output = Path(output).resolve()
    if not primary.is_file():
        raise FileNotFoundError(primary)
    if primary.suffix.lower() != ".swp":
        raise ValueError("Input must be a .swp file")
    if re.search(
        r"-part\d+\.swp$",
        primary.name,
        re.IGNORECASE
    ):
        raise ValueError(
            "Specify the primary swap file, "
            "not a continuation segment"
        )
    with primary.open("rb") as f:
        header = f.read(SWAP_HEADER_SIZE)
    if len(header) != SWAP_HEADER_SIZE:
        raise ValueError("Incomplete swap header")
    if not header.startswith(b"egamiwar"):
        raise ValueError(
            "Unrecognized PixInsight swap signature"
        )
    # Known Float32 and UInt16 grayscale/RGB swap layouts.
    sample_format, bytes_per_sample = swap_sample_format(header)
    width, height, channels = struct.unpack_from(
        "<III", header, 0x1c
    )
    thread_count = struct.unpack_from(
        "<Q", header, 0x68
    )[0]
    if not (
        1 <= width <= 200000
        and 1 <= height <= 200000
        and channels in (1, 3)
        and 1 <= thread_count <= 256
    ):
        raise ValueError(
            "Unsupported dimensions or thread count"
        )
    # Discover all parallel swap components.
    parts = [primary]
    for i in range(1, thread_count):
        part = primary.with_name(
            f"{primary.stem}-part{i:03d}.swp"
        )
        if not part.is_file():
            raise FileNotFoundError(
                f"Missing swap segment: {part}"
            )
        parts.append(part)
    expected_bytes = width * height * channels * bytes_per_sample
    actual_bytes = sum(
        part.stat().st_size -
        (SWAP_HEADER_SIZE if i == 0 else 0)
        for i, part in enumerate(parts)
    )
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Pixel data size mismatch: "
            f"{actual_bytes:,} vs {expected_bytes:,}"
        )
    if output == primary or output in parts:
        raise ValueError("Output path cannot be a source swap file")
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )
    print(f"Image dimensions: {width} x {height}")
    print(f"Parallel swap segments: {thread_count}")
    print(f"Pixel data: {expected_bytes:,} bytes")
    xml = create_xml(
        width,
        height,
        expected_bytes,
        channels=channels,
        sample_format=sample_format
    )
    if len(xml) + 16 > ATTACHMENT_OFFSET:
        raise ValueError(
            "XISF header exceeds reserved space"
        )
    dest = output.open("xb")
    try:
        with dest:
            # XISF binary preamble
            dest.write(
                XISF_SIGNATURE +
                struct.pack("<II", len(xml), 0)
            )
            # XML header
            dest.write(xml)
            # Pad to image attachment position
            padding = (
                ATTACHMENT_OFFSET - 16 - len(xml)
            )
            dest.write(b"\0" * padding)
            # Copy channel planes in XISF order.
            #
            # Important for RGB: PixInsight's parallel swap files store
            # one image strip per segment, with R, G and B blocks INSIDE
            # each segment. Concatenating whole RGB segments produces
            # visible horizontal stripes and misaligned colour channels.
            #
            # For XISF planar storage, concatenate every segment's R block,
            # then every segment's G block, then every segment's B block.
            # For mono, this is equivalent to the original sequential copy.
            #
            # The size of each RGB segment must be divisible into
            # channel strips. Mono segments have a single channel and
            # can be concatenated even when their boundaries split rows.
            channel_bytes = []
            for index, part in enumerate(parts):
                pixel_bytes = part.stat().st_size - (
                    SWAP_HEADER_SIZE if index == 0 else 0
                )
                if pixel_bytes < 0 or pixel_bytes % (channels * bytes_per_sample):
                    raise ValueError(
                        f"Cannot split channel blocks: {part.name}"
                    )
                # Segments can end partway through an image row for both
                # monochrome and RGB images. For RGB, each segment still
                # holds equally sized R, G, and B blocks. Split those blocks
                # on complete pixel-sample boundaries, not image rows.
                # The divisibility check above ensures this is possible.
                channel_bytes.append(pixel_bytes // channels)

            for channel in range(channels):
                print(
                    f"Writing {'RGB'[channel] if channels == 3 else 'Gray'} "
                    f"channel ({channel + 1}/{channels})",
                    flush=True
                )
                for index, part in enumerate(parts):
                    print(
                        f"  Segment {index + 1}/{len(parts)}: {part.name}",
                        flush=True
                    )
                    with part.open("rb") as src:
                        offset = (
                            (SWAP_HEADER_SIZE if index == 0 else 0)
                            + channel * channel_bytes[index]
                        )
                        src.seek(offset)
                        remaining = channel_bytes[index]
                        while remaining:
                            block = src.read(min(CHUNK_SIZE, remaining))
                            if not block:
                                raise EOFError(
                                    f"Unexpected end of segment: {part.name}"
                                )
                            dest.write(block)
                            remaining -= len(block)
        # Verify the resulting XISF structure.
        validate(output)
    except BaseException:
        # Remove incomplete output.
        # Never modify the source swap files.
        output.unlink(missing_ok=True)
        raise
    print(f"Recovery completed: {output}")

# --------------------------------------------------
# Batch recovery of a swap directory
# --------------------------------------------------


def original_image_identifier(primary):
    """Recover the image identifier from the PixInsight swap *filename*.

    Example: ~PI~TOKEN~MasterLight_Ha_c.swp -> MasterLight_Ha_c
    This name is not taken from embedded XISF metadata; there is no XISF yet.
    """
    match = re.fullmatch(r"~PI~([^~]+)~(.+)\.swp", primary.name, re.IGNORECASE)
    if not match:
        return None, None
    return match.group(2), match.group(1)



def safe_image_name(name):
    """Remove Windows-invalid path characters; retain useful image identifiers."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = name.strip(' .')
    return name[:160] or 'UnnamedImage'



def inspect_swap(primary):
    """Validate a whole group before adding it to the batch queue."""
    with primary.open('rb') as f:
        header = f.read(SWAP_HEADER_SIZE)
    if len(header) != SWAP_HEADER_SIZE or not header.startswith(b'egamiwar'):
        raise ValueError('Invalid or truncated PixInsight swap header')
    sample_format, bytes_per_sample = swap_sample_format(header)
    width, height, channels = struct.unpack_from('<III', header, 0x1c)
    thread_count = struct.unpack_from('<Q', header, 0x68)[0]
    if not (1 <= width <= 200000 and 1 <= height <= 200000
            and channels in (1, 3) and 1 <= thread_count <= 256):
        raise ValueError(f'Unsupported geometry/channels/threads: {width}x{height}, channels={channels}, threads={thread_count}')
    parts = [primary] + [primary.with_name(
        f'{primary.stem}-part{i:03d}.swp') for i in range(1, thread_count)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise ValueError('Missing continuation segment(s): ' + ', '.join(missing[:3]))
    expected = width * height * channels * bytes_per_sample
    actual = sum(p.stat().st_size - (SWAP_HEADER_SIZE if i == 0 else 0)
                 for i, p in enumerate(parts))
    if actual != expected:
        raise ValueError(f'Pixel data size mismatch: {actual:,} != {expected:,}')
    for i, part in enumerate(parts):
        pixel_bytes = part.stat().st_size - (SWAP_HEADER_SIZE if i == 0 else 0)
        if pixel_bytes < 0 or pixel_bytes % (channels * bytes_per_sample):
            raise ValueError(f'Invalid channel data size: {part.name}')
    return width, height, channels, len(parts), expected, sample_format



def recover_all(swap_directory, output_directory, dry_run=False):
    """Find primary files, recover each supported group, continue on errors."""
    swap_directory = Path(swap_directory).resolve()
    output_directory = Path(output_directory).resolve()
    if not swap_directory.is_dir():
        raise NotADirectoryError(swap_directory)
    if swap_directory == output_directory:
        raise ValueError('Output directory must be separate from swap directory')
    if not dry_run:
        output_directory.mkdir(parents=True, exist_ok=True)
    # Only scan this directory; numbered -partNNN.swp files are not primaries.
    primaries = sorted(
        (p for p in swap_directory.iterdir()
         if p.is_file() and p.suffix.lower() == '.swp'
         and not re.search(r'-part\d+\.swp$', p.name, re.IGNORECASE)),
        key=lambda p: p.name.lower()
    )
    print(f'Found {len(primaries)} candidate primary swap file(s).')
    recovered = skipped = failed = 0
    reserved_names = set()
    for index, primary in enumerate(primaries, 1):
        print(f'\n[{index}/{len(primaries)}] {primary.name}')
        identifier, token = original_image_identifier(primary)
        if identifier is None:
            print('  SKIP: filename does not match ~PI~TOKEN~ImageId.swp')
            skipped += 1
            continue
        try:
            width, height, channels, part_count, pixel_bytes, sample_format = inspect_swap(primary)
        except (OSError, ValueError, struct.error) as exc:
            print(f'  SKIP: {exc}')
            skipped += 1
            continue
        base = 'Recovered_' + safe_image_name(identifier)
        candidate = base + '.xisf'
        # Use the session token when identifiers collide; never overwrite.
        if candidate.lower() in reserved_names or (output_directory / candidate).exists():
            candidate = base + '_' + safe_image_name(token) + '.xisf'
        suffix = 2
        stem = Path(candidate).stem
        while (candidate.lower() in reserved_names or
               (output_directory / candidate).exists()):
            candidate = f'{stem}_{suffix}.xisf'
            suffix += 1
        reserved_names.add(candidate.lower())
        destination = output_directory / candidate
        print(f'  {width}x{height} {sample_format} {"Gray" if channels == 1 else "RGB"}, {part_count} segment(s), '
              f'{pixel_bytes:,} pixel bytes')
        print(f'  Output: {destination}')
        if dry_run:
            recovered += 1
            continue
        # A temporary file ensures an interrupted conversion never appears
        # under its final .xisf name. The original swap files stay untouched.
        try:
            with tempfile.TemporaryDirectory(prefix=".recover_", dir=output_directory) as work:
                temporary = Path(work) / "image.partial"
                recover(primary, temporary)
                # Both operations fail if another process claimed the name.
                if os.name == "nt":
                    temporary.rename(destination)
                else:
                    os.link(temporary, destination)
            recovered += 1
            print(f'  SUCCESS: {destination.name}')
        except Exception as exc:
            failed += 1
            print(f'  FAILED: {type(exc).__name__}: {exc}')
    print(f'\nBatch summary: {recovered} '
          f'{"eligible" if dry_run else "recovered"}, '
          f'{skipped} skipped, {failed} failed.')
    return 1 if failed else 0



# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main(argv=None):
    """Run the CLI; return a process exit status."""
    parser = argparse.ArgumentParser(
        description=(
            "Recover PixInsight parallel swap files to XISF 1.0. "
            "Use a COPY of an inactive swap directory."
        ),
        epilog=(
            "Examples:\n"
            "  python pi_swap_to_xisf.py primary.swp recovered.xisf\n"
            "  python pi_swap_to_xisf.py --all swap_backup --output-dir recovered --dry-run\n"
            "  python pi_swap_to_xisf.py --all swap_backup --output-dir recovered\n"
            "  python pi_swap_to_xisf.py --validate recovered.xisf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "primary", nargs="?", help="Primary PixInsight .swp file"
    )
    parser.add_argument(
        "output", nargs="?", help="Output XISF file"
    )
    parser.add_argument(
        "--repair-header", metavar="XISF",
        help="Repair a legacy recovery header in place (back up the XISF first)"
    )
    parser.add_argument(
        "--validate", metavar="XISF",
        help="Check a recovered XISF header and its uncompressed pixel attachment"
    )
    parser.add_argument(
        "--all", metavar="SWAP_DIR",
        help="Recover supported primary swap files from a directory"
    )
    parser.add_argument(
        "--output-dir", metavar="DIRECTORY",
        help="Destination directory for batch-recovered XISF images"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview eligible batch files and names without writing"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    args = parser.parse_args(argv)
    if args.all:
        if not args.output_dir:
            parser.error("--all requires --output-dir")
        if args.primary or args.output or args.repair_header or args.validate:
            parser.error("Batch mode cannot be combined with other operations")
    elif args.dry_run or args.output_dir:
        parser.error("--dry-run and --output-dir require --all")
    elif sum(bool(v) for v in (args.repair_header, args.validate)) + bool(args.primary or args.output) > 1:
        parser.error("Select only one operation")
    elif bool(args.primary) != bool(args.output):
        parser.error("Individual recovery requires both primary and output paths")

    try:
        if args.all:
            return recover_all(args.all, args.output_dir, args.dry_run)
        if args.repair_header:
            repair_header(args.repair_header)
        elif args.validate:
            validate(args.validate)
        elif args.primary and args.output:
            recover(args.primary, args.output)
        else:
            parser.print_help()
    except (OSError, ValueError, EOFError, ET.ParseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
