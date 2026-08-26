-- Secure API-key storage migration for api_tok.
-- Run this once in the Supabase SQL Editor before deploying the new backend.
-- The existing plaintext api_key column is intentionally retained temporarily
-- because the backend performs an automatic per-row migration on first access.

alter table public.api_tok
  add column if not exists api_key_hash text,
  add column if not exists api_key_encrypted text;

create unique index if not exists api_tok_api_key_hash_idx
  on public.api_tok (api_key_hash)
  where api_key_hash is not null;

-- After all legacy rows have been migrated and verified, run:
-- alter table public.api_tok drop column api_key;

-- IMPORTANT: api_key_encrypted is recoverable only by the backend using
-- API_KEY_ENCRYPTION_SECRET. Do not expose that secret to Supabase clients.
