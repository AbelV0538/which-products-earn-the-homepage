-- OPTIONAL profile (scans event_params): are multi-item view_item events one product or several? (rule R2)
WITH v AS (
  SELECT
    (SELECT LOWER(ep.value.string_value) FROM UNNEST(event_params) AS ep WHERE ep.key = 'page_title' LIMIT 1) AS page_title,
    ARRAY(
      SELECT LOWER(REGEXP_REPLACE(TRIM(x.item_name), r'\s+', ' '))
      FROM UNNEST(items) AS x WITH OFFSET AS o
      WHERE IF(TRIM(x.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(x.item_name)) IS NOT NULL
      ORDER BY o
    ) AS names
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'view_item'
)
SELECT
  COUNT(*) AS view_events,
  COUNTIF(ARRAY_LENGTH(names) > 1) AS multi_item_events,
  COUNTIF(ARRAY_LENGTH(names) > 1 AND (SELECT COUNT(DISTINCT n) FROM UNNEST(names) AS n) = 1) AS multi_item_same_name,
  COUNTIF(ARRAY_LENGTH(names) > 0 AND STRPOS(IFNULL(page_title, ''), names[SAFE_OFFSET(0)]) > 0) AS first_item_in_title,
  COUNTIF(EXISTS(SELECT 1 FROM UNNEST(names) AS n WHERE STRPOS(IFNULL(page_title, ''), n) > 0)) AS any_item_in_title
FROM v
