-- Profile: item_id <-> item_name cardinality and purchase coverage by id vs by name (rule R1).
WITH r AS (
  SELECT
    e.event_name,
    IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id)) AS item_id,
    LOWER(REGEXP_REPLACE(IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)), r'\s+', ' ')) AS name_key
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e, UNNEST(e.items) AS it
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND e.event_name IN ('view_item', 'add_to_cart', 'purchase')
),
pairs AS (
  SELECT item_id, name_key, COUNT(*) AS n
  FROM r
  WHERE item_id IS NOT NULL AND name_key IS NOT NULL
  GROUP BY 1, 2
),
by_id AS (SELECT item_id, COUNT(*) AS names, SUM(n) AS n FROM pairs GROUP BY 1),
by_name AS (SELECT name_key, COUNT(*) AS ids, SUM(n) AS n FROM pairs GROUP BY 1),
vi AS (SELECT DISTINCT item_id FROM r WHERE event_name = 'view_item' AND item_id IS NOT NULL),
vn AS (SELECT DISTINCT name_key FROM r WHERE event_name = 'view_item' AND name_key IS NOT NULL),
purch AS (
  SELECT vi.item_id IS NOT NULL AS id_viewed, vn.name_key IS NOT NULL AS name_viewed
  FROM r AS p
  LEFT JOIN vi ON p.item_id = vi.item_id
  LEFT JOIN vn ON p.name_key = vn.name_key
  WHERE p.event_name = 'purchase'
)
SELECT
  (SELECT COUNT(*) FROM by_id) AS distinct_ids,
  (SELECT COUNT(*) FROM by_name) AS distinct_names,
  (SELECT COUNTIF(names > 1) FROM by_id) AS ids_multi_name,
  (SELECT COUNTIF(ids > 1) FROM by_name) AS names_multi_id,
  (SELECT SAFE_DIVIDE(SUM(IF(ids > 1, n, 0)), SUM(n)) FROM by_name) AS row_share_multi_id_names,
  (SELECT SAFE_DIVIDE(SUM(IF(names > 1, n, 0)), SUM(n)) FROM by_id) AS row_share_multi_name_ids,
  (SELECT COUNT(*) FROM purch) AS purchase_rows,
  (SELECT COUNTIF(id_viewed) FROM purch) AS purchase_rows_id_viewed,
  (SELECT COUNTIF(name_viewed) FROM purch) AS purchase_rows_name_viewed,
  (SELECT ARRAY_AGG(STRUCT(name_key, ids, n) ORDER BY n DESC LIMIT 25) FROM by_name WHERE ids > 1) AS top_multi_id_names,
  (SELECT ARRAY_AGG(STRUCT(item_id, names, n) ORDER BY n DESC LIMIT 25) FROM by_id WHERE names > 1) AS top_multi_name_ids
