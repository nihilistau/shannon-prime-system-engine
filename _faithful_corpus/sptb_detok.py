"""sptb_detok.py <tok.sp-tokenizer> <id> [<id> ...] — detokenize ids via the SPTB vocab
(scans for the SPTB magic to skip any outer SPTK header; GPT-2 byte-level BPE inverse)."""
import struct, sys

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

U2B = {v: k for k, v in bytes_to_unicode().items()}

def load_vocab(path):
    blob = open(path, "rb").read()
    off = blob.find(b"SPTB")
    if off < 0: raise SystemExit("no SPTB magic")
    pos = off + 4
    type_id, vocab_size, n_merges = struct.unpack_from("<III", blob, pos); pos += 12
    vocab = []
    for _ in range(vocab_size):
        (ln,) = struct.unpack_from("<I", blob, pos); pos += 4
        vocab.append(blob[pos:pos + ln].decode("utf-8", "replace")); pos += ln
    return type_id, vocab

def detok(vocab, ids, type_id):
    out = bytearray()
    for i in ids:
        t = vocab[i] if 0 <= i < len(vocab) else f"<oov:{i}>"
        if type_id == 0:  # sentencepiece
            out += t.replace("▁", " ").encode()
        else:             # byte-level BPE
            out += bytes(U2B.get(c, ord("?") & 0xFF) if U2B.get(c) is not None else 63 for c in t)
    return out.decode("utf-8", "replace")

if __name__ == "__main__":
    type_id, vocab = load_vocab(sys.argv[1])
    ids = [int(x) for x in sys.argv[2:]]
    print(f"type_id={type_id} vocab={len(vocab)}")
    print(detok(vocab, ids, type_id))
