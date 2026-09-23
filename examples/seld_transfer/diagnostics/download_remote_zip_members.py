"""Download a few members from a remote Zip64 archive using HTTP Range."""

import argparse
import binascii
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def fetch_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": "bytes={}-{}".format(start, end)})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError("range length mismatch: {}-{} got {}".format(start, end, len(data)))
    return data


def parse_central_directory(tail, archive_size):
    eocd = tail.rfind(b"PK\x05\x06")
    zip64 = tail.rfind(b"PK\x06\x06")
    if eocd < 0 or zip64 < 0:
        raise RuntimeError("Zip64 end records not found")
    _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack_from("<4sQ2H2I2Q2Q", tail, zip64)
    tail_start = archive_size - len(tail)
    cd = tail[cd_offset - tail_start: cd_offset - tail_start + cd_size]
    if len(cd) != cd_size:
        raise RuntimeError("central directory is not fully contained in tail")
    entries = {}
    offset = 0
    central_format = "<4s6H3L5H2L"
    central_size = struct.calcsize(central_format)
    while offset < len(cd):
        header = struct.unpack_from(central_format, cd, offset)
        if header[0] != b"PK\x01\x02":
            raise RuntimeError("invalid central directory entry at {}".format(offset))
        compressed_size = header[8]
        uncompressed_size = header[9]
        name_len, extra_len, comment_len = header[10:13]
        local_offset = header[16]
        name_start = offset + central_size
        name = cd[name_start:name_start + name_len].decode("utf-8")
        extra = cd[name_start + name_len:name_start + name_len + extra_len]
        if compressed_size == 0xffffffff or uncompressed_size == 0xffffffff or local_offset == 0xffffffff:
            extra_offset = 0
            values = []
            while extra_offset + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, extra_offset)
                payload = extra[extra_offset + 4:extra_offset + 4 + size]
                if tag == 0x0001:
                    values = list(struct.unpack("<{}Q".format(len(payload) // 8), payload))
                    break
                extra_offset += 4 + size
            value_index = 0
            if uncompressed_size == 0xffffffff:
                uncompressed_size = values[value_index]
                value_index += 1
            if compressed_size == 0xffffffff:
                compressed_size = values[value_index]
                value_index += 1
            if local_offset == 0xffffffff:
                local_offset = values[value_index]
        entries[name] = {
            "compression": header[4],
            "crc32": header[7],
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }
        offset = name_start + name_len + extra_len + comment_len
    return entries


def extract_member(url, entry):
    local = fetch_range(url, entry["local_offset"], entry["local_offset"] + 29)
    fields = struct.unpack("<4s5H3L2H", local)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError("invalid local header")
    name_len, extra_len = fields[9:11]
    data_start = entry["local_offset"] + 30 + name_len + extra_len
    compressed = fetch_range(url, data_start, data_start + entry["compressed_size"] - 1)
    if entry["compression"] == 0:
        raw = compressed
    elif entry["compression"] == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError("unsupported compression method {}".format(entry["compression"]))
    if len(raw) != entry["uncompressed_size"]:
        raise RuntimeError("uncompressed size mismatch")
    if (binascii.crc32(raw) & 0xffffffff) != entry["crc32"]:
        raise RuntimeError("CRC mismatch")
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive-size", type=int, required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    tail = Path(args.tail).read_bytes()
    entries = parse_central_directory(tail, args.archive_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for member in args.member:
        if member not in entries:
            raise RuntimeError("member not found: {}".format(member))
        raw = extract_member(args.url, entries[member])
        output = output_dir / Path(member).name
        output.write_bytes(raw)
        manifest.append({"member": member, "output": str(output), **entries[member]})
    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
