-- Profile: device split (cross-check vs CSV: desktop 2,498,330 / mobile 1,704,069 / tablet 93,185).
SELECT
  LOWER(IFNULL(device.category, 'unknown')) AS device_category,
  COUNT(*) AS events,
  COUNTIF(event_name = 'view_item') AS view_item_events,
  COUNTIF(event_name = 'purchase') AS purchase_events
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY 1
ORDER BY events DESC
