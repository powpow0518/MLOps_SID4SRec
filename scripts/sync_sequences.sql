-- sync_sequences.sql
-- Run this after bulk ingestion to align sequences with current MAX(id) values.
-- This ensures POST /user and POST /item generate non-conflicting IDs.
--
-- Usage (inside the postgres container or via psql):
--   psql $DATABASE_URL -f scripts/sync_sequences.sql

SELECT setval('user_id_seq',     (SELECT COALESCE(MAX(user_id),     0) FROM "user"));
SELECT setval('category_id_seq', (SELECT COALESCE(MAX(category_id), 0) FROM category));
SELECT setval('brand_id_seq',    (SELECT COALESCE(MAX(brand_id),    0) FROM brand));
SELECT setval('item_id_seq',     (SELECT COALESCE(MAX(item_id),     0) FROM item));
