-- Secure API-key storage migration for api_tok.
-- Run this once in the Supabase SQL Editor before deploying the backend.
--
-- The old api_key column is no longer used to store credentials. It is kept
-- temporarily only so old rows can be migrated, and must therefore allow NULL.

alter table public.api_tok
  add column if not exists api_key_hash text,
  add column if not exists api_key_encrypted text;

-- Secure records deliberately set the legacy column to NULL.
alter table public.api_tok
  alter column api_key drop not null;

-- The old schema used VARCHAR(50) for owner. Email addresses can be longer,
-- so use text to prevent valid new accounts from failing during provisioning.
alter table public.api_tok
  alter column owner type text using owner::text;

create unique index if not exists api_tok_api_key_hash_idx
  on public.api_tok (api_key_hash)
  where api_key_hash is not null;

-- Existing plaintext rows can be migrated by the backend on access.
-- After ALL rows have been verified as migrated, remove plaintext storage:
-- alter table public.api_tok drop column api_key;

-- IMPORTANT: api_key_encrypted is recoverable only by the backend using
-- API_KEY_ENCRYPTION_SECRET. Never expose that secret to Supabase clients.
