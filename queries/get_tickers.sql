SELECT DISTINCT
       name AS ticker_id,
       company AS company_name,
       sector AS sector_name,
       industry AS industry_name
FROM dbr.tickers 
WHERE is_defunct = 0