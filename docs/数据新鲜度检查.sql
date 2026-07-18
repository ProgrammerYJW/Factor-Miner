-- ============================================================================
-- 聚源数据库(JYDB) 数据新鲜度检查脚本
-- 用途：核实各关键表数据实际更新到哪一天（官方口径 vs 实际入库）
-- 用法：整段复制到 SSMS / Rider 数据库控制台执行，全部为只读 SELECT
-- 说明：第一段用代表性个股/指数做探针（有索引支撑，秒级返回）；
--       第二段直接数全市场在几个关键日期的截面行数（最有说服力）。
--       注意 QT_TradingDayNew 是交易日历表，预填了未来日期，
--       它显示到 2026 年不代表行情数据更新到 2026 年。
-- 生成：FactorMiner 项目 2026-07-17
-- ============================================================================

-- 【第一段】各关键表最新数据日期（代表性探针）
SELECT * FROM (
    SELECT N'01 QT_DailyQuote 日行情(浦发600000)' AS 检查项,
           CONVERT(varchar(10), MAX(q.TradingDay), 120) AS 最新日期
      FROM dbo.QT_DailyQuote q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'02 LC_STIBDailyQuote 科创板行情(金山办公688111)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.LC_STIBDailyQuote q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '688111' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'03 QT_StockPerformance 行情表现(600000)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.QT_StockPerformance q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'04 QT_PerformanceData 后复权价/涨跌停标志(600000)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.QT_PerformanceData q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'05 LC_DIndicesForValuation 估值指标(600000)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.LC_DIndicesForValuation q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'06 QT_IndexQuote 指数行情(沪深300)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.QT_IndexQuote q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '000300' AND s.SecuCategory = 4
    UNION ALL
    SELECT N'07 QT_AdjustingFactor 复权因子(600000,除权除息日)',
           CONVERT(varchar(10), MAX(q.ExDiviDate), 120)
      FROM dbo.QT_AdjustingFactor q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'08 LC_SpecialTrade ST状态变更(全表)',
           CONVERT(varchar(10), MAX(SpecialTradeTime), 120)
      FROM dbo.LC_SpecialTrade
    UNION ALL
    SELECT N'09 MT_TradingDetail 融资融券(600000)',
           CONVERT(varchar(10), MAX(q.TradingDay), 120)
      FROM dbo.MT_TradingDetail q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'10 QT_TradingCapitalFlow 资金流向(600000)',
           CONVERT(varchar(10), MAX(q.TradingDate), 120)
      FROM dbo.QT_TradingCapitalFlow q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'11 LC_SHSZHSCHoldings 北向持股(600000)',
           CONVERT(varchar(10), MAX(q.EndDate), 120)
      FROM dbo.LC_SHSZHSCHoldings q JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'12 LC_MainIndexNew 财务指标·最大报告期(600000)',
           CONVERT(varchar(10), MAX(m.EndDate), 120)
      FROM dbo.LC_MainIndexNew m JOIN dbo.SecuMain s ON s.CompanyCode = m.CompanyCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'13 LC_MainIndexNew 财务指标·最大发布日(600000)',
           CONVERT(varchar(10), MAX(m.InfoPublDate), 120)
      FROM dbo.LC_MainIndexNew m JOIN dbo.SecuMain s ON s.CompanyCode = m.CompanyCode
     WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
    UNION ALL
    SELECT N'14 QT_TradingDayNew 交易日历(预填未来,仅供对照)',
           CONVERT(varchar(10), MAX(TradingDate), 120)
      FROM dbo.QT_TradingDayNew
     WHERE SecuMarket = 83
) t
ORDER BY 检查项;

-- 【第二段】全市场截面行数核对：这些日期当天到底有多少只股票有行情
-- （若某日行数为 0 或明显偏少，说明该日无数据/数据不全）
SELECT CONVERT(varchar(10), TradingDay, 120) AS 交易日,
       COUNT(*)                              AS 当日有行情的证券数
FROM dbo.QT_DailyQuote
WHERE TradingDay IN ('2024-12-30', '2024-12-31', '2025-03-31', '2025-06-30',
                     '2025-08-29', '2025-09-04', '2025-09-05', '2025-09-08')
GROUP BY TradingDay
ORDER BY TradingDay;
