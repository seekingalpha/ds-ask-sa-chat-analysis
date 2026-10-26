-- WITH processed_chat_tbl AS (

-- SELECT DISTINCT chat_id
-- FROM ds.production__ask_sa_sentiment_followup_msg_analysis

-- ),
    
-- user_input_tbl AS (

-- SELECT DISTINCT
--        start_time_ts,
--        user_id,
--        chat_id,
--        last_human_input AS msg,
--        chat_input_index,
--        'user' AS input_output_type
       
-- FROM dbr.langsmith 
-- WHERE -- is_root 
--       chat_input_index = 1
--       AND status IN ('success', 'DEFAULT')
--       AND chat_id NOT IN (SELECT chat_id FROM processed_chat_tbl)
--       AND date_ >= CURRENT_DATE - 3
-- -- ORDER BY chat_id, user_id, chat_input_index ASC

-- ),

-- output_raw_tbl AS (

-- SELECT DISTINCT
--        start_time_ts,
--        user_id,
--        chat_id,
--        trace_id,
--        CAST(msg AS VARCHAR(MAX)) AS outputs,
--        MAX(start_time_ts) OVER (PARTITION BY chat_id, trace_id) AS last_output_ts
-- FROM dbr.langsmith 
-- WHERE status IN ('success', 'DEFAULT')
--       AND chat_id IN (SELECT DISTINCT chat_id FROM user_input_tbl)
--       -- AND is_root = False
--       AND chat_input_index > 1
--       -- AND chat_id NOT IN (SELECT chat_id FROM processed_chat_tbl)
--       AND outputs NOT LIKE '%_end_%'
--       AND outputs NOT LIKE '%finish_reason%'
--       AND date_ >= CURRENT_DATE - 3

-- ),

-- chat_output_tbl AS (

-- SELECT start_time_ts,
--        user_id,
--        chat_id,
--       -- trace_id,
--       COALESCE(
--         -- generations[0][0].text
--         json_extract_path_text(
--           json_extract_array_element_text(
--             json_extract_array_element_text(
--               json_extract_path_text(outputs, 'generations', true), 0, true
--             ), 0, true
--           ),
--           'text', true
--         ),
    
--         -- messages[0].content  (works for [{"content":"..."}] but can error if messages[0] is an array)
--         json_extract_path_text(
--           json_extract_array_element_text(
--             json_extract_path_text(outputs, 'messages', true), 0, true
--           ),
--           'content', true
--         ),
    
--         -- messages[1].content  (works for [["user",".."], {"content":"..."}])
--         json_extract_path_text(
--           json_extract_array_element_text(
--             json_extract_path_text(outputs, 'messages', true), 1, true
--           ),
--           'content', true
--         ),
    
--         -- output
--         json_extract_path_text(outputs, 'output', true),
--         outputs
--       ) AS msg,
--        -1 AS chat_input_index,
--        'ai' AS input_output_type
-- FROM output_raw_tbl
-- WHERE last_output_ts = start_time_ts

-- ),

-- chat_output_clean_tbl AS (

-- SELECT DISTINCT
--        start_time_ts,
--        user_id,
--        chat_id,
--       -- chat_input_index,
--       -- input_output_type
--       -- msg
--       LISTAGG(LEFT(msg, 2000)) AS msg
-- FROM chat_output_tbl
-- WHERE msg != ''
-- GROUP BY 1, 2, 3

-- ),

-- union_tbl AS (

-- SELECT *
-- FROM user_input_tbl
-- UNION
-- SELECT start_time_ts,
--        user_id,
--        chat_id,
--        msg,
--        -1 AS chat_input_index,
--        'ai' AS input_output_type
-- FROM chat_output_clean_tbl

-- )

-- SELECT *
-- FROM union_tbl
-- -- WHERE chat_id = 183
-- ORDER BY chat_id, user_id, start_time_ts ASC


WITH processed_chat_tbl AS (

SELECT DISTINCT chat_id
FROM ds.production__ask_sa_sentiment_followup_msg_analysis

),
    
user_input_tbl AS (

SELECT DISTINCT
       start_time_ts,
       trace_id,
       user_id,
       chat_id,
       last_human_input AS msg,
       chat_input_index,
       'user' AS input_output_type
       
FROM dbr.langfuse 
WHERE -- is_root 
      chat_input_index >= 1
      AND status IN ('success', 'DEFAULT')
      AND chat_id NOT IN (SELECT chat_id FROM processed_chat_tbl)
      AND date_ >= CURRENT_DATE - 3
      -- AND date_ = '2026-07-14'::date
      AND COALESCE(project_id, tenant_id) = 'cmq6o9n7y0001xp07otd74m4b'
ORDER BY chat_id, user_id, chat_input_index ASC 

),

output_raw_tbl AS (

SELECT DISTINCT
       o.start_time_ts,
       o.user_id,
       o.chat_id,
       o.trace_id,
       CAST(o.msg AS VARCHAR(MAX)) AS outputs,
       MAX(o.start_time_ts) OVER (PARTITION BY o.chat_id, o.trace_id) AS last_output_ts
FROM dbr.langfuse AS o
INNER JOIN user_input_tbl AS i ON o.user_id = i.user_id
                                  AND o.chat_id = i.chat_id
                                  AND o.trace_id = i.trace_id
WHERE o.status IN ('success', 'DEFAULT')
      -- AND chat_id IN (SELECT DISTINCT chat_id FROM user_input_tbl)
      -- AND is_root = False
      AND o.chat_input_index IS NULL
      -- AND chat_id NOT IN (SELECT chat_id FROM processed_chat_tbl)
      -- AND outputs NOT LIKE '%_end_%'
      -- AND outputs NOT LIKE '%finish_reason%'
      AND date_ >= CURRENT_DATE - 3
      -- AND o.date_ = '2026-07-14'::date
      AND COALESCE(o.project_id, o.tenant_id) = 'cmq6o9n7y0001xp07otd74m4b'

),

chat_output_tbl AS (

SELECT start_time_ts,
       user_id,
       chat_id,
       trace_id,
       COALESCE(
         -- {"role": "assistant", "content": "..."}  (current format)
         NULLIF(json_extract_path_text(outputs, 'content', true), ''),

         -- generations[0][0].text
         NULLIF(
           json_extract_path_text(
             json_extract_array_element_text(
               json_extract_array_element_text(
                 json_extract_path_text(outputs, 'generations', true), 0, true
               ), 0, true
             ),
             'text', true
           ), ''),

         -- messages[0].content
         NULLIF(
           json_extract_path_text(
             json_extract_array_element_text(
               json_extract_path_text(outputs, 'messages', true), 0, true
             ),
             'content', true
           ), ''),

         -- messages[1].content
         NULLIF(
           json_extract_path_text(
             json_extract_array_element_text(
               json_extract_path_text(outputs, 'messages', true), 1, true
             ),
             'content', true
           ), ''),

         -- output
         NULLIF(json_extract_path_text(outputs, 'output', true), ''),

         -- unknown format: raw JSON
         outputs
       ) AS msg,
       -1 AS chat_input_index,
       'ai' AS input_output_type
FROM output_raw_tbl
WHERE last_output_ts = start_time_ts

),

chat_output_clean_tbl AS (

SELECT start_time_ts,
       trace_id,
       user_id,
       chat_id,
       -- keep head + tail: the Python side clips to 4000 chars (head + tail) itself,
       -- the 8000 cap only keeps LISTAGG below its 64KB limit
       LISTAGG(
         CASE WHEN LEN(msg) > 8000 THEN LEFT(msg, 4000) || ' ... ' || RIGHT(msg, 4000) ELSE msg END,
         ' '
       ) WITHIN GROUP (ORDER BY start_time_ts) AS msg
FROM chat_output_tbl
WHERE msg != ''
GROUP BY 1, 2, 3, 4

),

union_tbl AS (

SELECT *
FROM user_input_tbl
UNION
SELECT start_time_ts,
       trace_id,
       user_id,
       chat_id,
       msg,
      -1 AS chat_input_index,
      'ai' AS input_output_type
FROM chat_output_clean_tbl

)

SELECT *
FROM union_tbl
-- WHERE chat_id = 904446
ORDER BY chat_id, user_id, start_time_ts ASC