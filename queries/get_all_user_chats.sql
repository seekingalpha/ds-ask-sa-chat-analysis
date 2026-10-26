WITH ds_tbl AS (
    SELECT DISTINCT
           chat_id,
           trace_id
    FROM ds.production__ask_sa_message_entity_extraction
)

SELECT DISTINCT
       l.start_time_ts AS ts,
       l.user_id,
       l.chat_id,
       l.trace_id,
       l.last_human_input AS msg
FROM dbr.langfuse AS l
WHERE l.chat_input_index = 1
      AND l.status IN ('success', 'DEFAULT')
      AND l.user_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM ds_tbl d WHERE d.chat_id = l.chat_id)
      AND NOT EXISTS (SELECT 1 FROM ds_tbl d WHERE d.trace_id = l.trace_id)
      AND l.date_ >= CURRENT_DATE - 4
      AND COALESCE(l.project_id, l.tenant_id) = 'cmq6o9n7y0001xp07otd74m4b'
