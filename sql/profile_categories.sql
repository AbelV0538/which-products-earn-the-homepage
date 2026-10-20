-- Profile: raw category values with volume and example products (drives the category mapping, rule R3).
SELECT
  it.item_category,
  it.item_category2,
  COUNTIF(e.event_name = 'view_item') AS view_rows,
  COUNTIF(e.event_name = 'purchase') AS purchase_rows,
  COUNT(DISTINCT IF(TRIM(it.item_id) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_id))) AS items,
  ARRAY_AGG(DISTINCT IF(TRIM(it.item_name) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(it.item_name)) IGNORE NULLS LIMIT 5) AS example_names
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` AS e, UNNEST(e.items) AS it
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  AND e.event_name IN ('view_item', 'purchase')
GROUP BY 1, 2
ORDER BY view_rows DESC
LIMIT 500
