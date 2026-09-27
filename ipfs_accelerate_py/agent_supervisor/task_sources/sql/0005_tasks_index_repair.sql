-- DuckDB ART indexes that include the mutable VARCHAR status column fail
-- UPDATE tasks SET status=... with:
--   Failed to delete all rows from index. Only deleted 0 out of 1 rows
-- and invalidate the owner connection. Keep goal/ordinal indexes without status.
DROP INDEX IF EXISTS tasks_status_idx;
DROP INDEX IF EXISTS tasks_goal_idx;
-- On pinned DuckDB 1.5.5, recreating a persisted populated index under
-- its old name aborts at COMMIT. Use a distinct replacement name.
CREATE INDEX tasks_goal_replacement_idx ON tasks(goal_cid);
CREATE INDEX tasks_ordinal_idx ON tasks(ordinal);
