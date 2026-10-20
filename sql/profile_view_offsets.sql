-- OPTIONAL profile (scans event_params): is item offset 0 of a view_item event the product actually viewed? (rule R2)
-- For every product bought in a session, was it shown in that session's view_item events, and at offset 0?
-- If offset 0 were the viewed product, nearly every viewed purchase would have been seen there.
WITH v AS (
  SELECT
    user_pseudo_id,
    (SELECT ep.value.int_value FROM UNNEST(event_params) AS ep WHERE ep.key = 'ga_session_id') AS sid,
    LOWER(REGEXP_REPLACE(TRIM(it.item_name), r'\s+', ' ')) AS name_key,
    off
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`, UNNEST(items) AS it WITH OFFSET AS off
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'view_item'
    AND IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) IS NOT NULL
),
p AS (
  SELECT DISTINCT
    user_pseudo_id,
    (SELECT ep.value.int_value FROM UNNEST(event_params) AS ep WHERE ep.key = 'ga_session_id') AS sid,
    LOWER(REGEXP_REPLACE(TRIM(it.item_name), r'\s+', ' ')) AS name_key
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`, UNNEST(items) AS it
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'purchase'
    AND IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) IS NOT NULL
),
agg AS (
  SELECT
    p.user_pseudo_id,
    p.sid,
    p.name_key,
    LOGICAL_OR(v.off = 0) AS seen_at_0,
    LOGICAL_OR(v.off IS NOT NULL) AS seen_any,
    MIN(v.off) AS min_off
  FROM p
  LEFT JOIN v
    ON p.user_pseudo_id = v.user_pseudo_id AND p.sid = v.sid AND p.name_key = v.name_key
  GROUP BY 1, 2, 3
)
SELECT
  COUNT(*) AS purchased_items,
  COUNTIF(seen_any) AS seen_in_session_any_offset,
  COUNTIF(seen_at_0) AS seen_in_session_at_offset0,
  APPROX_QUANTILES(min_off, 10) AS min_offset_deciles
FROM agg
