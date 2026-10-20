-- Profile: how many items does each event type carry? (decides the view metric, rule R2)
SELECT
  event_name,
  COUNT(*) AS events,
  COUNTIF(ARRAY_LENGTH(items) > 0) AS events_with_items,
  AVG(ARRAY_LENGTH(items)) AS avg_items,
  AVG(IF(ARRAY_LENGTH(items) > 0, ARRAY_LENGTH(items), NULL)) AS avg_items_nonempty,
  MAX(ARRAY_LENGTH(items)) AS max_items,
  APPROX_QUANTILES(ARRAY_LENGTH(items), 100)[OFFSET(99)] AS p99_items
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY event_name
ORDER BY events DESC
