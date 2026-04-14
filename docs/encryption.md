# Database Encryption (v3.3j+)

The `.nmdb` files are encrypted with HMAC-SHA256-based stream cipher (encrypt-then-MAC).
Replaced the old gzip+header obfuscation which agents could trivially reverse-engineer.

## Key Derivation

**Source file:** `src/saas_bench/db_protection.py` (internal repo only)

The key is derived at runtime via PBKDF2 from:
- **Passphrase:** split across `_KP` list (8 fragments joined at runtime)
- **Salt:** split across `_KS` list (4 fragments joined at runtime)
- **Algorithm:** PBKDF2-HMAC-SHA256, 100k iterations -> 64-byte key (32 enc + 32 mac)

## Wire Format

`nonce(16) || HMAC-SHA256(32) || ciphertext`

- No magic bytes, no detectable structure
- File looks like random bytes (chi-squared test passes)
- HMAC-then-decrypt: wrong key is rejected before any decryption

## Where the Key Lives (NOT in agent workspace)

| Location | Has Key? | Notes |
|----------|----------|-------|
| `src/saas_bench/db_protection.py` | YES | Internal repo source code |
| PyInstaller binary (`novamind-server`) | YES (compiled) | In compiled bytecode, not trivially extractable |
| `monitor/push_data.py` | YES (imports) | Our dashboard data pusher |
| Agent workspace | NO | Only has `.nmdb` file + compiled binary |
| Public repo (`zlab-princeton/run-ceobench`) | NO | Only has compiled binary |

## What the Agent Has Access To

- The `.nmdb` file (encrypted, looks like random bytes)
- The `novamind-server` binary (key compiled in, not trivially extractable)
- The `novamind-operation` CLI (doesn't contain key -- it's a thin wrapper)

## To Decrypt Manually (internal repo only)

```python
from saas_bench.db_protection import load_session_db
conn = load_session_db(Path("path/to/world.nmdb"))
# conn is a sqlite3.Connection to in-memory DB
```

## Test Results (2026-04-14)

All 7 tests passed:
1. Raw bytes -- no SQLite header, no CREATE TABLE, no NOVAMIND magic
2. sqlite3.connect() -- "file is not a database"
3. gzip.decompress() -- "Not a gzipped file"
4. gzip magic scan -- coincidental 1f8b at random offset, decompress fails
5. Wrong key -- "HMAC verification failed"
6. Entropy -- chi-sq=220.1, all 256 byte values present, indistinguishable from random
7. Correct key -- decrypted successfully, 44 tables accessible
