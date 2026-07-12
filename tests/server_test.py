# Drive a generated alpacc binary (C or CUDA backend) in --server mode and
# transcode between the u64-BE test protocol and the native server protocol.
#
# The server protocol uses the grammar-native types (terminal_t,
# production_t, index_t) in host byte order; the batch/test protocol is
# u64 BE.  This helper queries `<binary> --layout` for the native type
# sizes, builds native request frames from the u64-BE test inputs, and
# transcodes the native responses back to the u64-BE test format (with
# num_tests prefix) so the result can be diffed against batch output or
# checked with `alpacc test compare`.
#
# Usage: python3 server_test.py <binary> <inputs_file> <output_file> <kind>
# where <kind> is "lexer", "parser", or "both".

import sys, struct, subprocess

def u64be(v): return struct.pack(">Q", v)
def u64le(v): return struct.pack("<Q", v)
def read_u64be(data, off): return struct.unpack_from(">Q", data, off)[0], off + 8
def rd(data, off, size):
    return int.from_bytes(data[off:off+size], "little"), off + size

binary, inputs_file, output_file, kind = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

layout = {}
for line in subprocess.check_output([binary, "--layout"]).decode().splitlines():
    key, _, val = line.partition("=")
    layout[key.strip()] = int(val)

terminal_size = layout["terminal_t"]
production_size = layout.get("production_t", 0)
index_size = layout.get("index_t", 8)

with open(inputs_file, "rb") as f:
    data = f.read()

off = 0
num_tests, off = read_u64be(data, off)

proc = subprocess.Popen([binary, "--server"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL)

# Build native request frames from the u64-BE test inputs.
parts = []
for _ in range(num_tests):
    n, off = read_u64be(data, off)
    if kind == "parser":
        toks = struct.unpack_from(f">{n}Q", data, off)
        off += 8 * n
        payload = b"".join(t.to_bytes(terminal_size, "little") for t in toks)
    else:
        payload = data[off:off + n]
        off += n
    content = u64le(n) + payload
    parts.append(u64le(len(content)))
    parts.append(content)
frames = b"".join(parts)

# communicate() interleaves stdin writes with stdout reads, avoiding a
# deadlock when the server's output outruns the pipe buffer.
raw, _ = proc.communicate(frames)
if proc.returncode != 0:
    sys.stderr.write(f"server exit {proc.returncode}\n")
    sys.exit(1)

# Transcode the native responses back to the u64-BE test format.
parts = [u64be(num_tests)]
roff = 0
for _ in range(num_tests):
    valid = raw[roff]; roff += 1
    parts.append(bytes([valid]))
    if valid == 0:
        continue
    count, roff = rd(raw, roff, 8)
    parts.append(u64be(count))
    for _ in range(count):
        if kind == "parser":
            v, roff = rd(raw, roff, production_size)
            parts.append(u64be(v))
        elif kind == "lexer":
            t, roff = rd(raw, roff, terminal_size)
            s, roff = rd(raw, roff, index_size)
            e, roff = rd(raw, roff, index_size)
            parts.append(u64be(t) + u64be(s) + u64be(e))
        else:
            is_term = raw[roff]; roff += 1
            p, roff = rd(raw, roff, index_size)
            i, roff = rd(raw, roff, production_size)
            s, roff = rd(raw, roff, index_size)
            e, roff = rd(raw, roff, index_size)
            parts.append(bytes([is_term]) + u64be(p) + u64be(i) + u64be(s) + u64be(e))
out = b"".join(parts)

if roff != len(raw):
    sys.stderr.write("error: trailing bytes in server output\n")
    sys.exit(1)

with open(output_file, "wb") as f:
    f.write(out)
