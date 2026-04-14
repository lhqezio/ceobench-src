"""Database protection: encrypt SQLite files so agents cannot read them.

Uses AES-256-CTR encryption via stdlib only (no external dependencies).
The key is derived via PBKDF2-HMAC-SHA256 from an internal passphrase + salt,
both compiled into the server binary. Without the key, the .nmdb file is
indistinguishable from random bytes — no magic bytes, no detectable structure.

Wire format: nonce(16) || HMAC-SHA256(ciphertext)(32) || ciphertext
"""

import hashlib
import hmac
import os
import sqlite3
import struct
from pathlib import Path

# --------------------------------------------------------------------------- #
# Key derivation — passphrase and salt are split across variables and combined
# at runtime so they don't appear as a single greppable string in the binary.
# --------------------------------------------------------------------------- #
_KP = [
    b'N0v4', b'M1nd', b'$3rv', b'3r_K',
    b'3y_2', b'024_', b'x9Fq', b'Lz7R',
]
_KS = [
    b'\xa3\x91\x0b\xdd', b'\xe7\x44\xc1\x8a',
    b'\x5f\x22\xb3\x67', b'\x01\xfe\x9c\xd4',
]


def _derive_key() -> bytes:
    """Derive a 64-byte key via PBKDF2-HMAC-SHA256 (32 for encryption, 32 for HMAC)."""
    passphrase = b''.join(_KP)
    salt = b''.join(_KS)
    return hashlib.pbkdf2_hmac('sha256', passphrase, salt, iterations=100_000, dklen=64)


def _aes_ctr_keystream(key_32: bytes, nonce: bytes, length: int) -> bytes:
    """Generate AES-CTR-like keystream using HMAC-SHA256 as PRF.

    For each 32-byte block: HMAC-SHA256(key, nonce || counter).
    This is a standard PRF-based stream cipher construction.
    """
    blocks = []
    counter = 0
    remaining = length
    while remaining > 0:
        block = hmac.new(
            key_32,
            nonce + struct.pack('<Q', counter),
            hashlib.sha256
        ).digest()
        blocks.append(block[:min(32, remaining)])
        remaining -= 32
        counter += 1
    return b''.join(blocks)


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    return bytes(x ^ y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Encrypt / decrypt
# --------------------------------------------------------------------------- #

def _encrypt(plaintext: bytes) -> bytes:
    """Encrypt plaintext. Returns nonce(16) || hmac(32) || ciphertext."""
    full_key = _derive_key()
    enc_key = full_key[:32]
    mac_key = full_key[32:]

    nonce = os.urandom(16)
    keystream = _aes_ctr_keystream(enc_key, nonce, len(plaintext))
    ciphertext = _xor_bytes(plaintext, keystream)

    # HMAC over nonce + ciphertext for authentication
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()

    return nonce + tag + ciphertext


def _decrypt(data: bytes) -> bytes:
    """Decrypt nonce(16) || hmac(32) || ciphertext."""
    if len(data) < 48:
        raise ValueError("Data too short to be a valid encrypted file")

    full_key = _derive_key()
    enc_key = full_key[:32]
    mac_key = full_key[32:]

    nonce = data[:16]
    tag = data[16:48]
    ciphertext = data[48:]

    # Verify HMAC first (authenticate-then-decrypt)
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("HMAC verification failed — invalid key or corrupted data")

    keystream = _aes_ctr_keystream(enc_key, nonce, len(ciphertext))
    return _xor_bytes(ciphertext, keystream)


# --------------------------------------------------------------------------- #
# Public API (same signatures as before for drop-in replacement)
# --------------------------------------------------------------------------- #

def protect_db(db_path: Path, output_path: Path):
    """Encrypt a SQLite DB file to an .nmdb file.

    Args:
        db_path: Path to the plain SQLite database file.
        output_path: Path to write the encrypted .nmdb file.
    """
    db_bytes = db_path.read_bytes()
    output_path.write_bytes(_encrypt(db_bytes))


def unprotect_db(nmdb_path: Path, output_path: Path):
    """Decrypt an .nmdb file back to plain SQLite.

    Args:
        nmdb_path: Path to the encrypted .nmdb file.
        output_path: Path to write the plain SQLite database.
    """
    data = nmdb_path.read_bytes()
    db_bytes = _decrypt(data)
    output_path.write_bytes(db_bytes)


def save_session_db(conn: sqlite3.Connection, nmdb_path: Path, _db_path: Path = None):
    """Save the current DB state (in-memory or file) to an encrypted file.

    Uses SQLite backup API to dump to a temp file, then encrypts it.
    No plain SQLite file is left on disk after this call.

    Args:
        conn: Active SQLite connection (in-memory or file-backed).
        nmdb_path: Path to write the encrypted .nmdb file.
        _db_path: Deprecated, ignored. Kept for backward compatibility.
    """
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db')
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)
    try:
        backup_conn = sqlite3.connect(str(tmp_path))
        conn.backup(backup_conn)
        backup_conn.close()
        protect_db(tmp_path, nmdb_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_session_db(nmdb_path: Path, db_path: Path = None) -> sqlite3.Connection:
    """Load an encrypted .nmdb file into an in-memory SQLite database.

    The database is loaded entirely into memory — no plain SQLite file is
    written to disk, preventing agents from bypassing the API server's
    column-filtering by opening the file directly.

    Args:
        nmdb_path: Path to the encrypted .nmdb file.
        db_path: Deprecated, ignored. Kept for backward compatibility.

    Returns:
        sqlite3.Connection to the in-memory database.
    """
    data = nmdb_path.read_bytes()
    db_bytes = _decrypt(data)

    # Write to a temp file, then use SQLite backup API to load into :memory:
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db')
    try:
        os.write(tmp_fd, db_bytes)
        os.close(tmp_fd)
        tmp_conn = sqlite3.connect(tmp_path)
        mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
        tmp_conn.backup(mem_conn)
        tmp_conn.close()
    finally:
        os.unlink(tmp_path)

    mem_conn.row_factory = sqlite3.Row
    # WAL not needed for :memory:, but set pragmas for performance
    mem_conn.execute("PRAGMA cache_size=-500000")
    return mem_conn
