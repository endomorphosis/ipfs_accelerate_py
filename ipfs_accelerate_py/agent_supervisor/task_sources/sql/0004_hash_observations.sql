-- Operational observations only: TTL reuse is not proof or execution authority.
-- Only the existing exclusive Quack owner mutates these bounded cache rows.
CREATE TABLE hash_observations (
    cache_key VARCHAR PRIMARY KEY,
    identity_json VARCHAR NOT NULL,
    owner_generation VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    principal VARCHAR NOT NULL,
    lease_token VARCHAR NOT NULL,
    fence BIGINT NOT NULL,
    claimed_at_ms BIGINT NOT NULL,
    lease_expires_ms BIGINT NOT NULL,
    ttl_ms BIGINT NOT NULL,
    sha256 VARCHAR NOT NULL,
    expires_at_ms BIGINT NOT NULL
);
CREATE INDEX hash_observations_expiry_idx
    ON hash_observations(expires_at_ms);
