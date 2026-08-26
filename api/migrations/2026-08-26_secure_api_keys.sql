-- Secure API-key storage migration for api_tok.
-- Run this once in the Supabase SQL Editor.
--
-- The legacy api_key column is retained temporarily so existing records can
-- be migrated. It MUST be nullable because secure records no longer store a
-- plaintext API key.

alter table public.api_tok
  add column if not exists api_key_hash text,
  add column if not exists api_key_encrypted text;

alter table public.api_tok
  alter column api_key drop not null;

create unique index if not exists api_tok_api_key_hash_idx
  on public.api_tok (api_key_hash)
  where api_key_hash is not null;

-- After all legacy rows have been migrated and verified, run:
-- alter table public.api_tok drop column api_key;

-- IMPORTANT: api_key_encrypted is recoverable only by the backend using
-- API_KEY_ENCRYPTION_SECRET. Never expose that secret to Supabase clients.
