-- Run this in Supabase SQL Editor before deploying secure API-key provisioning.
-- New secure records store only api_key_hash and api_key_encrypted.
-- The legacy plaintext column must therefore accept NULL.

alter table public.api_tok
  alter column api_key drop not null;

create unique index if not exists api_tok_api_key_hash_idx
  on public.api_tok (api_key_hash)
  where api_key_hash is not null;
