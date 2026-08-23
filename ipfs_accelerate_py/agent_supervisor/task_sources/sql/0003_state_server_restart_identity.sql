-- Admit sequential state-owner generations within one live process birth.
--
-- ``state_servers`` retains one row per server generation, while
-- ``server_epochs`` retains its corresponding lifecycle interval.  A process
-- can stop and restart the owner without changing its OS birth identity, so
-- birth identity alone cannot be unique across durable generations.
DROP INDEX state_servers_birth_uidx;
CREATE UNIQUE INDEX state_servers_birth_generation_uidx
    ON state_servers(process_birth_id, generation);
