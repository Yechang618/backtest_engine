# 训练字段字典：final selected model panel

## 生成口径

- 来源 trace: `/data/data_process/4.15_revision/model_training_stage1_11_selected_panel/field_decision_trace_all.csv`
- 输出字段数: `459`
- 模型输入候选数: `452`
- 已强制排除: `APPLICABLE_MISSING_RATE`、全部 `*_HIGH_MISSING_FLAG`
- 说明: `FWD_RET_5D_Z_P01_P99` 是监督学习标签，保留在训练面板中，但不作为模型输入特征。

## 角色统计

| 角色 | 中文含义 | 字段数 |
|---|---|---:|
| `control_mask` | 训练/交易控制掩码 | 3 |
| `financial_or_fundamental` | 财务/基本面字段 | 152 |
| `label` | 训练标签 | 1 |
| `mask` | 字段掩码 | 196 |
| `metadata` | 元数据 | 3 |
| `technical_factor` | 量价技术因子 | 104 |

## 全部保留字段

| 序号 | 字段名 | 中文含义 | 角色 | 是否模型输入 | 保留原因 |
|---:|---|---|---|---:|---|
| 1 | `S_INFO_WINDCODE` | Wind 股票代码，股票唯一标识 | 元数据 | 0 | `keep_metadata_not_model_input` |
| 2 | `TRADE_DT` | 交易日期，YYYYMMDD | 元数据 | 0 | `keep_metadata_not_model_input` |
| 3 | `SW_L1_CODE` | 申万一级行业代码 | 元数据 | 0 | `keep_metadata_not_model_input` |
| 4 | `Breadth_global` | 全市场宽度指标 | 量价技术因子 | 1 | `keep_technical_factor` |
| 5 | `Breadth_industry` | 所属行业宽度指标 | 量价技术因子 | 1 | `keep_technical_factor` |
| 6 | `ACCT_PAYABLE__APPLICABLE_MASK` | 应付账款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 7 | `ACCT_PAYABLE__MISSING_MASK` | 应付账款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 8 | `ACCT_RCV__APPLICABLE_MASK` | 应收账款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 9 | `ACCT_RCV__MISSING_MASK` | 应收账款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 10 | `AMORT_INTANG_ASSETS__APPLICABLE_MASK` | 无形资产摊销是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 11 | `AMORT_INTANG_ASSETS__MISSING_MASK` | 无形资产摊销在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 12 | `AMORT_LT_DEFERRED_EXP__APPLICABLE_MASK` | 长期待摊费用摊销是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 13 | `AMORT_LT_DEFERRED_EXP__MISSING_MASK` | 长期待摊费用摊销在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 14 | `BONDS_PAYABLE__APPLICABLE_MASK` | 应付债券是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 15 | `BONDS_PAYABLE__MISSING_MASK` | 应付债券在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 16 | `BORROW_CENTRAL_BANK__APPLICABLE_MASK` | 向中央银行借款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 17 | `BORROW_CENTRAL_BANK__MISSING_MASK` | 向中央银行借款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 18 | `Breadth_global__APPLICABLE_MASK` | 全市场宽度指标是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 19 | `Breadth_global__MISSING_MASK` | 全市场宽度指标在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 20 | `Breadth_industry__APPLICABLE_MASK` | 所属行业宽度指标是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 21 | `Breadth_industry__MISSING_MASK` | 所属行业宽度指标在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 22 | `CAP_RSRV__APPLICABLE_MASK` | 资本公积是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 23 | `CAP_RSRV__MISSING_MASK` | 资本公积在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 24 | `CASH_CASH_EQU_BEG_PERIOD__APPLICABLE_MASK` | 期初现金及现金等价物余额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 25 | `CASH_CASH_EQU_BEG_PERIOD__MISSING_MASK` | 期初现金及现金等价物余额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 26 | `CASH_CASH_EQU_END_PERIOD__APPLICABLE_MASK` | 期末现金及现金等价物余额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 27 | `CASH_CASH_EQU_END_PERIOD__MISSING_MASK` | 期末现金及现金等价物余额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 28 | `CASH_PAY_ACQ_CONST_FIOLTA__APPLICABLE_MASK` | 购建固定资产、无形资产和其他长期资产支付的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 29 | `CASH_PAY_ACQ_CONST_FIOLTA__MISSING_MASK` | 购建固定资产、无形资产和其他长期资产支付的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 30 | `CASH_PAY_BEH_EMPL__APPLICABLE_MASK` | 支付给职工以及为职工支付的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 31 | `CASH_PAY_BEH_EMPL__MISSING_MASK` | 支付给职工以及为职工支付的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 32 | `CASH_PAY_DIST_DPCP_INT_EXP__APPLICABLE_MASK` | 分配股利、利润或偿付利息支付的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 33 | `CASH_PAY_DIST_DPCP_INT_EXP__MISSING_MASK` | 分配股利、利润或偿付利息支付的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 34 | `CASH_PAY_GOODS_PURCH_SERV_REC__APPLICABLE_MASK` | 购买商品、接受劳务支付的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 35 | `CASH_PAY_GOODS_PURCH_SERV_REC__MISSING_MASK` | 购买商品、接受劳务支付的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 36 | `CASH_PREPAY_AMT_BORR__APPLICABLE_MASK` | 偿还债务支付的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 37 | `CASH_PREPAY_AMT_BORR__MISSING_MASK` | 偿还债务支付的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 38 | `CASH_RECP_BORROW__APPLICABLE_MASK` | 取得借款收到的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 39 | `CASH_RECP_BORROW__MISSING_MASK` | 取得借款收到的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 40 | `CASH_RECP_CAP_CONTRIB__APPLICABLE_MASK` | 吸收投资收到的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 41 | `CASH_RECP_CAP_CONTRIB__MISSING_MASK` | 吸收投资收到的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 42 | `CASH_RECP_DISP_WITHDRWL_INVEST__APPLICABLE_MASK` | 收回投资收到的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 43 | `CASH_RECP_DISP_WITHDRWL_INVEST__MISSING_MASK` | 收回投资收到的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 44 | `CASH_RECP_RETURN_INVEST__APPLICABLE_MASK` | 取得投资收益收到的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 45 | `CASH_RECP_RETURN_INVEST__MISSING_MASK` | 取得投资收益收到的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 46 | `CASH_RECP_SG_AND_RS__APPLICABLE_MASK` | 销售商品、提供劳务收到的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 47 | `CASH_RECP_SG_AND_RS__MISSING_MASK` | 销售商品、提供劳务收到的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 48 | `DEFERRED_TAX_ASSETS__APPLICABLE_MASK` | 递延所得税资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 49 | `DEFERRED_TAX_ASSETS__MISSING_MASK` | 递延所得税资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 50 | `DEFERRED_TAX_LIAB__APPLICABLE_MASK` | 递延所得税负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 51 | `DEFERRED_TAX_LIAB__MISSING_MASK` | 递延所得税负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 52 | `DEPR_FA_COGA_DPBA__APPLICABLE_MASK` | 固定资产折旧、油气资产折耗、生产性生物资产折旧是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 53 | `DEPR_FA_COGA_DPBA__MISSING_MASK` | 固定资产折旧、油气资产折耗、生产性生物资产折旧在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 54 | `DVD_PAYABLE__APPLICABLE_MASK` | 应付股利是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 55 | `DVD_PAYABLE__MISSING_MASK` | 应付股利在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 56 | `EFF_FX_FLU_CASH__APPLICABLE_MASK` | 汇率变动对现金及现金等价物的影响是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 57 | `EFF_FX_FLU_CASH__MISSING_MASK` | 汇率变动对现金及现金等价物的影响在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 58 | `EMPL_BEN_PAYABLE__APPLICABLE_MASK` | 应付职工薪酬是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 59 | `EMPL_BEN_PAYABLE__MISSING_MASK` | 应付职工薪酬在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 60 | `END_BAL_CASH__APPLICABLE_MASK` | 现金期末余额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 61 | `END_BAL_CASH__MISSING_MASK` | 现金期末余额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 62 | `FIX_ASSETS__APPLICABLE_MASK` | 固定资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 63 | `FIX_ASSETS__MISSING_MASK` | 固定资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 64 | `FIX_ASSETS_DISP__APPLICABLE_MASK` | 固定资产清理是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 65 | `FIX_ASSETS_DISP__MISSING_MASK` | 固定资产清理在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 66 | `GOODWILL__APPLICABLE_MASK` | 商誉是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 67 | `GOODWILL__MISSING_MASK` | 商誉在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 68 | `IM_NET_CASH_FLOWS_OPER_ACT__APPLICABLE_MASK` | 间接法经营活动现金流量净额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 69 | `IM_NET_CASH_FLOWS_OPER_ACT__MISSING_MASK` | 间接法经营活动现金流量净额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 70 | `IM_NET_INCR_CASH_CASH_EQU__APPLICABLE_MASK` | 间接法现金及现金等价物净增加额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 71 | `IM_NET_INCR_CASH_CASH_EQU__MISSING_MASK` | 间接法现金及现金等价物净增加额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 72 | `INT_INC__APPLICABLE_MASK` | 利息收入是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 73 | `INT_INC__MISSING_MASK` | 利息收入在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 74 | `INT_PAYABLE__APPLICABLE_MASK` | 应付利息是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 75 | `INT_PAYABLE__MISSING_MASK` | 应付利息在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 76 | `INT_RCV__APPLICABLE_MASK` | 应收利息是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 77 | `INT_RCV__MISSING_MASK` | 应收利息在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 78 | `INVENTORIES__APPLICABLE_MASK` | 存货是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 79 | `INVENTORIES__MISSING_MASK` | 存货在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 80 | `LESS_BEG_BAL_CASH__APPLICABLE_MASK` | 现金期初余额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 81 | `LESS_BEG_BAL_CASH__MISSING_MASK` | 现金期初余额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 82 | `LESS_BEG_BAL_CASH_EQU__APPLICABLE_MASK` | LESS_BEG_BAL_CASH_EQU是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 83 | `LESS_BEG_BAL_CASH_EQU__MISSING_MASK` | LESS_BEG_BAL_CASH_EQU在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 84 | `LESS_FIN_EXP__APPLICABLE_MASK` | 财务费用是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 85 | `LESS_FIN_EXP__MISSING_MASK` | 财务费用在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 86 | `LESS_GERL_ADMIN_EXP__APPLICABLE_MASK` | 管理费用是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 87 | `LESS_GERL_ADMIN_EXP__MISSING_MASK` | 管理费用在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 88 | `LESS_OPER_COST__APPLICABLE_MASK` | 营业成本是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 89 | `LESS_OPER_COST__MISSING_MASK` | 营业成本在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 90 | `LESS_SELLING_DIST_EXP__APPLICABLE_MASK` | 销售费用是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 91 | `LESS_SELLING_DIST_EXP__MISSING_MASK` | 销售费用在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 92 | `LESS_TAXES_SURCHARGES_OPS__APPLICABLE_MASK` | 税金及附加是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 93 | `LESS_TAXES_SURCHARGES_OPS__MISSING_MASK` | 税金及附加在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 94 | `LOANS_OTH_BANKS__APPLICABLE_MASK` | 同业及其他金融机构拆入款项是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 95 | `LOANS_OTH_BANKS__MISSING_MASK` | 同业及其他金融机构拆入款项在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 96 | `LONG_TERM_EQY_INVEST__APPLICABLE_MASK` | 长期股权投资是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 97 | `LONG_TERM_EQY_INVEST__MISSING_MASK` | 长期股权投资在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 98 | `LT_BORROW__APPLICABLE_MASK` | 长期借款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 99 | `LT_BORROW__MISSING_MASK` | 长期借款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 100 | `MINORITY_INT__APPLICABLE_MASK` | 少数股东权益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 101 | `MINORITY_INT__MISSING_MASK` | 少数股东权益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 102 | `MINORITY_INT_INC__APPLICABLE_MASK` | 少数股东损益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 103 | `MINORITY_INT_INC__MISSING_MASK` | 少数股东损益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 104 | `MONETARY_CAP__APPLICABLE_MASK` | 货币资金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 105 | `MONETARY_CAP__MISSING_MASK` | 货币资金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 106 | `NET_CASH_FLOWS_FNC_ACT__APPLICABLE_MASK` | 筹资活动现金流量净额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 107 | `NET_CASH_FLOWS_FNC_ACT__MISSING_MASK` | 筹资活动现金流量净额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 108 | `NET_CASH_FLOWS_INV_ACT__APPLICABLE_MASK` | 投资活动现金流量净额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 109 | `NET_CASH_FLOWS_INV_ACT__MISSING_MASK` | 投资活动现金流量净额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 110 | `NET_CASH_FLOWS_OPER_ACT__APPLICABLE_MASK` | 经营活动现金流量净额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 111 | `NET_CASH_FLOWS_OPER_ACT__MISSING_MASK` | 经营活动现金流量净额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 112 | `NET_INCR_CASH_CASH_EQU__APPLICABLE_MASK` | 现金及现金等价物净增加额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 113 | `NET_INCR_CASH_CASH_EQU__MISSING_MASK` | 现金及现金等价物净增加额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 114 | `NET_PROFIT_EXCL_MIN_INT_INC__APPLICABLE_MASK` | 归属于母公司股东的净利润是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 115 | `NET_PROFIT_EXCL_MIN_INT_INC__MISSING_MASK` | 归属于母公司股东的净利润在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 116 | `NET_PROFIT_INCL_MIN_INT_INC__APPLICABLE_MASK` | 净利润，含少数股东损益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 117 | `NET_PROFIT_INCL_MIN_INT_INC__MISSING_MASK` | 净利润，含少数股东损益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 118 | `NON_CUR_LIAB_DUE_WITHIN_1Y__APPLICABLE_MASK` | 一年内到期的非流动负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 119 | `NON_CUR_LIAB_DUE_WITHIN_1Y__MISSING_MASK` | 一年内到期的非流动负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 120 | `NOTES_PAYABLE__APPLICABLE_MASK` | 应付票据是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 121 | `NOTES_PAYABLE__MISSING_MASK` | 应付票据在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 122 | `OIL_AND_NATURAL_GAS_ASSETS__APPLICABLE_MASK` | 油气资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 123 | `OIL_AND_NATURAL_GAS_ASSETS__MISSING_MASK` | 油气资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 124 | `OPER_PROFIT__APPLICABLE_MASK` | 营业利润是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 125 | `OPER_PROFIT__MISSING_MASK` | 营业利润在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 126 | `OPER_REV__APPLICABLE_MASK` | 营业收入是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 127 | `OPER_REV__MISSING_MASK` | 营业收入在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 128 | `OTHER_CASH_PAY_RAL_FNC_ACT__APPLICABLE_MASK` | 支付其他与筹资活动有关的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 129 | `OTHER_CASH_PAY_RAL_FNC_ACT__MISSING_MASK` | 支付其他与筹资活动有关的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 130 | `OTHER_CASH_PAY_RAL_OPER_ACT__APPLICABLE_MASK` | 支付其他与经营活动有关的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 131 | `OTHER_CASH_PAY_RAL_OPER_ACT__MISSING_MASK` | 支付其他与经营活动有关的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 132 | `OTHER_CASH_RECP_RAL_FNC_ACT__APPLICABLE_MASK` | 收到其他与筹资活动有关的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 133 | `OTHER_CASH_RECP_RAL_FNC_ACT__MISSING_MASK` | 收到其他与筹资活动有关的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 134 | `OTHER_CASH_RECP_RAL_INV_ACT__APPLICABLE_MASK` | 收到其他与投资活动有关的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 135 | `OTHER_CASH_RECP_RAL_INV_ACT__MISSING_MASK` | 收到其他与投资活动有关的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 136 | `OTHER_CASH_RECP_RAL_OPER_ACT__APPLICABLE_MASK` | 收到其他与经营活动有关的现金是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 137 | `OTHER_CASH_RECP_RAL_OPER_ACT__MISSING_MASK` | 收到其他与经营活动有关的现金在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 138 | `OTH_CUR_ASSETS__APPLICABLE_MASK` | 其他流动资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 139 | `OTH_CUR_ASSETS__MISSING_MASK` | 其他流动资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 140 | `OTH_CUR_LIAB__APPLICABLE_MASK` | 其他流动负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 141 | `OTH_CUR_LIAB__MISSING_MASK` | 其他流动负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 142 | `OTH_NON_CUR_ASSETS__APPLICABLE_MASK` | 其他非流动资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 143 | `OTH_NON_CUR_ASSETS__MISSING_MASK` | 其他非流动资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 144 | `OTH_NON_CUR_LIAB__APPLICABLE_MASK` | 其他非流动负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 145 | `OTH_NON_CUR_LIAB__MISSING_MASK` | 其他非流动负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 146 | `OTH_PAYABLE__APPLICABLE_MASK` | 其他应付款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 147 | `OTH_PAYABLE__MISSING_MASK` | 其他应付款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 148 | `PLUS_END_BAL_CASH_EQU__APPLICABLE_MASK` | PLUS_END_BAL_CASH_EQU是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 149 | `PLUS_END_BAL_CASH_EQU__MISSING_MASK` | PLUS_END_BAL_CASH_EQU在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 150 | `PRODUCTIVE_BIO_ASSETS__APPLICABLE_MASK` | 生产性生物资产是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 151 | `PRODUCTIVE_BIO_ASSETS__MISSING_MASK` | 生产性生物资产在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 152 | `PROV_NOM_RISKS__APPLICABLE_MASK` | 预计负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 153 | `PROV_NOM_RISKS__MISSING_MASK` | 预计负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 154 | `RD_EXPENSE__APPLICABLE_MASK` | 研发费用是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 155 | `RD_EXPENSE__MISSING_MASK` | 研发费用在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 156 | `STOT_CASH_INFLOWS_FNC_ACT__APPLICABLE_MASK` | 筹资活动现金流入小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 157 | `STOT_CASH_INFLOWS_FNC_ACT__MISSING_MASK` | 筹资活动现金流入小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 158 | `STOT_CASH_INFLOWS_INV_ACT__APPLICABLE_MASK` | 投资活动现金流入小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 159 | `STOT_CASH_INFLOWS_INV_ACT__MISSING_MASK` | 投资活动现金流入小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 160 | `STOT_CASH_INFLOWS_OPER_ACT__APPLICABLE_MASK` | 经营活动现金流入小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 161 | `STOT_CASH_INFLOWS_OPER_ACT__MISSING_MASK` | 经营活动现金流入小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 162 | `STOT_CASH_OUTFLOWS_FNC_ACT__APPLICABLE_MASK` | 筹资活动现金流出小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 163 | `STOT_CASH_OUTFLOWS_FNC_ACT__MISSING_MASK` | 筹资活动现金流出小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 164 | `STOT_CASH_OUTFLOWS_INV_ACT__APPLICABLE_MASK` | 投资活动现金流出小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 165 | `STOT_CASH_OUTFLOWS_INV_ACT__MISSING_MASK` | 投资活动现金流出小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 166 | `STOT_CASH_OUTFLOWS_OPER_ACT__APPLICABLE_MASK` | 经营活动现金流出小计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 167 | `STOT_CASH_OUTFLOWS_OPER_ACT__MISSING_MASK` | 经营活动现金流出小计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 168 | `ST_BONDS_PAYABLE__APPLICABLE_MASK` | 应付短期债券是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 169 | `ST_BONDS_PAYABLE__MISSING_MASK` | 应付短期债券在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 170 | `ST_BORROW__APPLICABLE_MASK` | 短期借款是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 171 | `ST_BORROW__MISSING_MASK` | 短期借款在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 172 | `SURPLUS_RSRV__APPLICABLE_MASK` | 盈余公积是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 173 | `SURPLUS_RSRV__MISSING_MASK` | 盈余公积在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 174 | `S_FA_EPS_BASIC__APPLICABLE_MASK` | 基本每股收益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 175 | `S_FA_EPS_BASIC__MISSING_MASK` | 基本每股收益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 176 | `S_FA_EPS_DILUTED__APPLICABLE_MASK` | 稀释每股收益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 177 | `S_FA_EPS_DILUTED__MISSING_MASK` | 稀释每股收益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 178 | `TAXES_SURCHARGES_PAYABLE__APPLICABLE_MASK` | 应交税费是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 179 | `TAXES_SURCHARGES_PAYABLE__MISSING_MASK` | 应交税费在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 180 | `TOT_ASSETS__APPLICABLE_MASK` | 资产总计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 181 | `TOT_ASSETS__MISSING_MASK` | 资产总计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 182 | `TOT_CUR_LIAB__APPLICABLE_MASK` | 流动负债合计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 183 | `TOT_CUR_LIAB__MISSING_MASK` | 流动负债合计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 184 | `TOT_LIAB__APPLICABLE_MASK` | 负债合计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 185 | `TOT_LIAB__MISSING_MASK` | 负债合计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 186 | `TOT_LIAB_SHRHLDR_EQY__APPLICABLE_MASK` | 负债和股东权益总计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 187 | `TOT_LIAB_SHRHLDR_EQY__MISSING_MASK` | 负债和股东权益总计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 188 | `TOT_NON_CUR_LIAB__APPLICABLE_MASK` | 非流动负债合计是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 189 | `TOT_NON_CUR_LIAB__MISSING_MASK` | 非流动负债合计在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 190 | `TOT_OPER_REV__APPLICABLE_MASK` | 营业总收入是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 191 | `TOT_OPER_REV__MISSING_MASK` | 营业总收入在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 192 | `TOT_PROFIT__APPLICABLE_MASK` | 利润总额是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 193 | `TOT_PROFIT__MISSING_MASK` | 利润总额在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 194 | `TOT_SHRHLDR_EQY_EXCL_MIN_INT__APPLICABLE_MASK` | 归属于母公司股东权益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 195 | `TOT_SHRHLDR_EQY_EXCL_MIN_INT__MISSING_MASK` | 归属于母公司股东权益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 196 | `TOT_SHRHLDR_EQY_INCL_MIN_INT__APPLICABLE_MASK` | 股东权益合计，含少数股东权益是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 197 | `TOT_SHRHLDR_EQY_INCL_MIN_INT__MISSING_MASK` | 股东权益合计，含少数股东权益在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 198 | `TRADABLE_FIN_LIAB__APPLICABLE_MASK` | 交易性金融负债是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 199 | `TRADABLE_FIN_LIAB__MISSING_MASK` | 交易性金融负债在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 200 | `UNDISTRIBUTED_PROFIT__APPLICABLE_MASK` | 未分配利润是否适用于该股票所属行业，1=适用，0=不适用 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 201 | `UNDISTRIBUTED_PROFIT__MISSING_MASK` | 未分配利润在适用情况下是否缺失，1=缺失或已填充，0=原始存在 | 字段掩码 | 1 | `force_keep_all_missing_applicable_masks` |
| 202 | `SUPER_TREND_DIRECTION` | SuperTrend 趋势指标，按复权价量构造的量价技术因子 | 量价技术因子 | 1 | `keep_technical_factor` |
| 203 | `UP_FRACTAL_FLAG` | 向上分形指标，按复权价量构造的量价技术因子 | 量价技术因子 | 1 | `keep_technical_factor` |
| 204 | `DOWN_FRACTAL_FLAG` | 向下分形指标，按复权价量构造的量价技术因子 | 量价技术因子 | 1 | `keep_technical_factor` |
| 205 | `TOT_OPER_REV_TTM_LAG_3Y_MISSING_FLAG` | 三年前滚动十二个月营业总收入缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 206 | `AVG_ACCT_RCV_MISSING_FLAG` | 平均应收账款缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 207 | `AVG_EQUITY_MISSING_FLAG` | 平均股东权益缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 208 | `AVG_INVENTORIES_MISSING_FLAG` | 平均存货缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 209 | `AVG_MONETARY_CAP_MISSING_FLAG` | 平均货币资金缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 210 | `EPS_SINGLE_Q_MISSING_FLAG` | 单季度每股收益缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 211 | `AMORT_INTANG_ASSETS_TTM_MISSING_FLAG` | 滚动十二个月无形资产摊销缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 212 | `AMORT_LT_DEFERRED_EXP_TTM_MISSING_FLAG` | 滚动十二个月长期待摊费用摊销缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 213 | `CASH_PAY_ACQ_CONST_FIOLTA_TTM_MISSING_FLAG` | 滚动十二个月资本开支相关现金流出缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 214 | `DEPR_FA_COGA_DPBA_TTM_MISSING_FLAG` | 滚动十二个月折旧缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 215 | `NET_CASH_FLOWS_OPER_ACT_TTM_MISSING_FLAG` | 滚动十二个月经营活动现金流量净额缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 216 | `LESS_OPER_COST_TTM_MISSING_FLAG` | 滚动十二个月营业成本缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 217 | `NET_PROFIT_TTM_MISSING_FLAG` | 滚动十二个月净利润缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 218 | `TOT_OPER_REV_TTM_MISSING_FLAG` | 滚动十二个月营业总收入缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 219 | `INTEREST_BEARING_LIAB_MISSING_FLAG` | 带息负债缺失填充标志，1=原始缺失/被补值，0=原始存在 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 220 | `PV_ADJ_OPEN_TO_PREVCLOSE_RET_MKT_Z` | PV_ADJ_OPEN_TO_PREVCLOSE_RET，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 221 | `PV_ADJ_HIGH_TO_PREVCLOSE_RET_MKT_Z` | PV_ADJ_HIGH_TO_PREVCLOSE_RET，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 222 | `PV_ADJ_LOW_TO_PREVCLOSE_RET_MKT_Z` | PV_ADJ_LOW_TO_PREVCLOSE_RET，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 223 | `PV_ADJ_CLOSE_RET_1D_MKT_Z` | PV_ADJ_CLOSE_RET_1D，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 224 | `PV_ADJ_INTRADAY_RET_MKT_Z` | PV_ADJ_INTRADAY_RET，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 225 | `PV_ADJ_HIGH_LOW_RANGE_MKT_Z` | PV_ADJ_HIGH_LOW_RANGE，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 226 | `PV_LIMIT_RET_FROM_PRECLOSE_MKT_Z` | PV_LIMIT_RET_FROM_PRECLOSE，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 227 | `PV_STOPPING_RET_FROM_PRECLOSE_MKT_Z` | PV_STOPPING_RET_FROM_PRECLOSE，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 228 | `PV_VOLUME_LOG_MKT_Z` | PV_VOLUME_LOG，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 229 | `PV_AMOUNT_LOG_MKT_Z` | PV_AMOUNT_LOG，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 230 | `PV_CAPITAL_LOG_MKT_Z` | PV_CAPITAL_LOG，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 231 | `BS_DAYS_SINCE_UPDATE_MKT_Z` | 距离上次资产负债表更新的交易日数，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 232 | `IC_DAYS_SINCE_UPDATE_MKT_Z` | 距离上次利润表更新的交易日数，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 233 | `CF_DAYS_SINCE_UPDATE_MKT_Z` | 距离上次现金流量表更新的交易日数，经全市场当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_financial_or_fundamental` |
| 234 | `ANCHOR_AGE_MKT_Z` | 锚定 VWAP 指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 235 | `DAYS_SINCE_LAST_UP_FRACTAL_MKT_Z` | DAYS_SINCE_LAST_UP_FRACTAL，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 236 | `DAYS_SINCE_LAST_DOWN_FRACTAL_MKT_Z` | DAYS_SINCE_LAST_DOWN_FRACTAL，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 237 | `MONETARY_CAP_IND_Z` | 货币资金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 238 | `ACCT_RCV_IND_Z` | 应收账款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 239 | `INT_RCV_IND_Z` | 应收利息，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 240 | `INVENTORIES_IND_Z` | 存货，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 241 | `OTH_CUR_ASSETS_IND_Z` | 其他流动资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 242 | `LONG_TERM_EQY_INVEST_IND_Z` | 长期股权投资，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 243 | `FIX_ASSETS_IND_Z` | 固定资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 244 | `FIX_ASSETS_DISP_IND_Z` | 固定资产清理，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 245 | `PRODUCTIVE_BIO_ASSETS_IND_Z` | 生产性生物资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 246 | `OIL_AND_NATURAL_GAS_ASSETS_IND_Z` | 油气资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 247 | `GOODWILL_IND_Z` | 商誉，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 248 | `DEFERRED_TAX_ASSETS_IND_Z` | 递延所得税资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 249 | `OTH_NON_CUR_ASSETS_IND_Z` | 其他非流动资产，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 250 | `TOT_ASSETS_IND_Z` | 资产总计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 251 | `ST_BORROW_IND_Z` | 短期借款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 252 | `BORROW_CENTRAL_BANK_IND_Z` | 向中央银行借款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 253 | `LOANS_OTH_BANKS_IND_Z` | 同业及其他金融机构拆入款项，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 254 | `TRADABLE_FIN_LIAB_IND_Z` | 交易性金融负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 255 | `NOTES_PAYABLE_IND_Z` | 应付票据，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 256 | `ACCT_PAYABLE_IND_Z` | 应付账款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 257 | `EMPL_BEN_PAYABLE_IND_Z` | 应付职工薪酬，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 258 | `TAXES_SURCHARGES_PAYABLE_IND_Z` | 应交税费，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 259 | `INT_PAYABLE_IND_Z` | 应付利息，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 260 | `DVD_PAYABLE_IND_Z` | 应付股利，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 261 | `OTH_PAYABLE_IND_Z` | 其他应付款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 262 | `ST_BONDS_PAYABLE_IND_Z` | 应付短期债券，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 263 | `NON_CUR_LIAB_DUE_WITHIN_1Y_IND_Z` | 一年内到期的非流动负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 264 | `OTH_CUR_LIAB_IND_Z` | 其他流动负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 265 | `TOT_CUR_LIAB_IND_Z` | 流动负债合计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 266 | `LT_BORROW_IND_Z` | 长期借款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 267 | `BONDS_PAYABLE_IND_Z` | 应付债券，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 268 | `DEFERRED_TAX_LIAB_IND_Z` | 递延所得税负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 269 | `OTH_NON_CUR_LIAB_IND_Z` | 其他非流动负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 270 | `TOT_NON_CUR_LIAB_IND_Z` | 非流动负债合计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 271 | `TOT_LIAB_IND_Z` | 负债合计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 272 | `CAP_RSRV_IND_Z` | 资本公积，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 273 | `SURPLUS_RSRV_IND_Z` | 盈余公积，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 274 | `UNDISTRIBUTED_PROFIT_IND_Z` | 未分配利润，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 275 | `PROV_NOM_RISKS_IND_Z` | 预计负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 276 | `MINORITY_INT_IND_Z` | 少数股东权益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 277 | `TOT_SHRHLDR_EQY_EXCL_MIN_INT_IND_Z` | 归属于母公司股东权益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 278 | `TOT_SHRHLDR_EQY_INCL_MIN_INT_IND_Z` | 股东权益合计，含少数股东权益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 279 | `TOT_LIAB_SHRHLDR_EQY_IND_Z` | 负债和股东权益总计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 280 | `CASH_RECP_SG_AND_RS_IND_Z` | 销售商品、提供劳务收到的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 281 | `OTHER_CASH_RECP_RAL_OPER_ACT_IND_Z` | 收到其他与经营活动有关的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 282 | `STOT_CASH_INFLOWS_OPER_ACT_IND_Z` | 经营活动现金流入小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 283 | `CASH_PAY_GOODS_PURCH_SERV_REC_IND_Z` | 购买商品、接受劳务支付的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 284 | `CASH_PAY_BEH_EMPL_IND_Z` | 支付给职工以及为职工支付的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 285 | `OTHER_CASH_PAY_RAL_OPER_ACT_IND_Z` | 支付其他与经营活动有关的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 286 | `STOT_CASH_OUTFLOWS_OPER_ACT_IND_Z` | 经营活动现金流出小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 287 | `NET_CASH_FLOWS_OPER_ACT_IND_Z` | 经营活动现金流量净额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 288 | `CASH_RECP_DISP_WITHDRWL_INVEST_IND_Z` | 收回投资收到的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 289 | `CASH_RECP_RETURN_INVEST_IND_Z` | 取得投资收益收到的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 290 | `OTHER_CASH_RECP_RAL_INV_ACT_IND_Z` | 收到其他与投资活动有关的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 291 | `STOT_CASH_INFLOWS_INV_ACT_IND_Z` | 投资活动现金流入小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 292 | `CASH_PAY_ACQ_CONST_FIOLTA_IND_Z` | 购建固定资产、无形资产和其他长期资产支付的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 293 | `STOT_CASH_OUTFLOWS_INV_ACT_IND_Z` | 投资活动现金流出小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 294 | `NET_CASH_FLOWS_INV_ACT_IND_Z` | 投资活动现金流量净额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 295 | `CASH_RECP_CAP_CONTRIB_IND_Z` | 吸收投资收到的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 296 | `CASH_RECP_BORROW_IND_Z` | 取得借款收到的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 297 | `OTHER_CASH_RECP_RAL_FNC_ACT_IND_Z` | 收到其他与筹资活动有关的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 298 | `STOT_CASH_INFLOWS_FNC_ACT_IND_Z` | 筹资活动现金流入小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 299 | `CASH_PREPAY_AMT_BORR_IND_Z` | 偿还债务支付的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 300 | `CASH_PAY_DIST_DPCP_INT_EXP_IND_Z` | 分配股利、利润或偿付利息支付的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 301 | `OTHER_CASH_PAY_RAL_FNC_ACT_IND_Z` | 支付其他与筹资活动有关的现金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 302 | `STOT_CASH_OUTFLOWS_FNC_ACT_IND_Z` | 筹资活动现金流出小计，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 303 | `NET_CASH_FLOWS_FNC_ACT_IND_Z` | 筹资活动现金流量净额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 304 | `EFF_FX_FLU_CASH_IND_Z` | 汇率变动对现金及现金等价物的影响，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 305 | `NET_INCR_CASH_CASH_EQU_IND_Z` | 现金及现金等价物净增加额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 306 | `CASH_CASH_EQU_BEG_PERIOD_IND_Z` | 期初现金及现金等价物余额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 307 | `CASH_CASH_EQU_END_PERIOD_IND_Z` | 期末现金及现金等价物余额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 308 | `DEPR_FA_COGA_DPBA_IND_Z` | 固定资产折旧、油气资产折耗、生产性生物资产折旧，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 309 | `AMORT_INTANG_ASSETS_IND_Z` | 无形资产摊销，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 310 | `AMORT_LT_DEFERRED_EXP_IND_Z` | 长期待摊费用摊销，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 311 | `IM_NET_CASH_FLOWS_OPER_ACT_IND_Z` | 间接法经营活动现金流量净额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 312 | `END_BAL_CASH_IND_Z` | 现金期末余额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 313 | `LESS_BEG_BAL_CASH_IND_Z` | 现金期初余额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 314 | `PLUS_END_BAL_CASH_EQU_IND_Z` | PLUS_END_BAL_CASH_EQU，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 315 | `LESS_BEG_BAL_CASH_EQU_IND_Z` | LESS_BEG_BAL_CASH_EQU，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 316 | `IM_NET_INCR_CASH_CASH_EQU_IND_Z` | 间接法现金及现金等价物净增加额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 317 | `TOT_OPER_REV_IND_Z` | 营业总收入，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 318 | `OPER_REV_IND_Z` | 营业收入，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 319 | `INT_INC_IND_Z` | 利息收入，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 320 | `LESS_OPER_COST_IND_Z` | 营业成本，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 321 | `LESS_TAXES_SURCHARGES_OPS_IND_Z` | 税金及附加，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 322 | `LESS_SELLING_DIST_EXP_IND_Z` | 销售费用，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 323 | `LESS_GERL_ADMIN_EXP_IND_Z` | 管理费用，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 324 | `LESS_FIN_EXP_IND_Z` | 财务费用，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 325 | `OPER_PROFIT_IND_Z` | 营业利润，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 326 | `TOT_PROFIT_IND_Z` | 利润总额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 327 | `NET_PROFIT_INCL_MIN_INT_INC_IND_Z` | 净利润，含少数股东损益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 328 | `NET_PROFIT_EXCL_MIN_INT_INC_IND_Z` | 归属于母公司股东的净利润，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 329 | `MINORITY_INT_INC_IND_Z` | 少数股东损益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 330 | `RD_EXPENSE_IND_Z` | 研发费用，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 331 | `INTEREST_BEARING_LIAB_IND_Z` | 带息负债，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 332 | `AVG_INVENTORIES_IND_Z` | 平均存货，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 333 | `AVG_ACCT_RCV_IND_Z` | 平均应收账款，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 334 | `AVG_EQUITY_IND_Z` | 平均股东权益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 335 | `AVG_MONETARY_CAP_IND_Z` | 平均货币资金，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 336 | `NET_PROFIT_TTM_IND_Z` | 滚动十二个月净利润，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 337 | `TOT_OPER_REV_TTM_IND_Z` | 滚动十二个月营业总收入，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 338 | `LESS_OPER_COST_TTM_IND_Z` | 滚动十二个月营业成本，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 339 | `TOT_OPER_REV_TTM_LAG_3Y_IND_Z` | 三年前滚动十二个月营业总收入，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 340 | `NET_CASH_FLOWS_OPER_ACT_TTM_IND_Z` | 滚动十二个月经营活动现金流量净额，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 341 | `CASH_PAY_ACQ_CONST_FIOLTA_TTM_IND_Z` | 滚动十二个月资本开支相关现金流出，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 342 | `DEPR_FA_COGA_DPBA_TTM_IND_Z` | 滚动十二个月折旧，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 343 | `AMORT_INTANG_ASSETS_TTM_IND_Z` | 滚动十二个月无形资产摊销，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 344 | `AMORT_LT_DEFERRED_EXP_TTM_IND_Z` | 滚动十二个月长期待摊费用摊销，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 345 | `S_FA_EPS_BASIC_IND_Z` | 基本每股收益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 346 | `S_FA_EPS_DILUTED_IND_Z` | 稀释每股收益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 347 | `EPS_SINGLE_Q_IND_Z` | 单季度每股收益，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 348 | `INVENTORY_TURNOVER_IND_Z` | 存货周转率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 349 | `ROE_TTM_IND_Z` | 滚动十二个月净资产收益率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 350 | `AR_TURNOVER_IND_Z` | 应收账款周转率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 351 | `CAPEX_TO_DA_IND_Z` | 资本开支 / 折旧摊销，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 352 | `CASHFLOW_RATIO_IND_Z` | 现金流量比率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 353 | `INTEREST_BEARING_LIAB_RATIO_IND_Z` | 带息负债率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 354 | `TOT_OPER_REV_3Y_GROWTH_IND_Z` | 营业总收入三年增长率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 355 | `EQUITY_MULTIPLIER_IND_Z` | 权益乘数，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 356 | `OCF_TO_INTEREST_BEARING_DEBT_IND_Z` | 经营现金流量净额 / 带息债务，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 357 | `CASH_TURNOVER_IND_Z` | 现金周转率，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 358 | `PE_TTM_IND_Z` | 市盈率 TTM，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 359 | `PS_TTM_IND_Z` | 市销率 TTM，经所属行业当日截面去极值/标准化后的特征 | 财务/基本面字段 | 1 | `keep_preferred_normalized_version__IND_Z` |
| 360 | `PPO_RAW_MKT_Z` | 百分比价格振荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 361 | `PPO_SIGNAL_MKT_Z` | 百分比价格振荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 362 | `PPO_HIST_MKT_Z` | 百分比价格振荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 363 | `ADX_14_MKT_Z` | 平均趋向指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 364 | `PLUS_DI_14_MKT_Z` | 正向趋向指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 365 | `MINUS_DI_14_MKT_Z` | 负向趋向指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 366 | `AROON_UP_25_MKT_Z` | Aroon 趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 367 | `AROON_DOWN_25_MKT_Z` | Aroon 趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 368 | `AROON_OSC_25_MKT_Z` | Aroon 趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 369 | `KAMA_GAP_MKT_Z` | 考夫曼自适应均线，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 370 | `KAMA_SLOPE_MKT_Z` | 考夫曼自适应均线，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 371 | `MFI_14_MKT_Z` | 资金流量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 372 | `OBV_REL_MKT_Z` | 能量潮指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 373 | `CMF_20_MKT_Z` | 蔡金货币流量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 374 | `FORCE_RAW_MKT_Z` | 强力指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 375 | `FORCE_EMA_2_MKT_Z` | 强力指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 376 | `FORCE_LOG_TANH_MKT_Z` | 强力指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 377 | `DC_POSITION_20_MKT_Z` | 唐奇安通道指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 378 | `DC_WIDTH_20_MKT_Z` | 唐奇安通道指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 379 | `IFT_RSI_14_MKT_Z` | 逆费舍尔变换 RSI，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 380 | `DEMARKER_14_MKT_Z` | DeMarker 需求供给衰竭指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 381 | `ADTM_23_MKT_Z` | 动态买卖气指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 382 | `RVI_10_MKT_Z` | 相对活力指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 383 | `RVI_SIGNAL_10_4_MKT_Z` | 相对活力指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 384 | `FISHER_10_MKT_Z` | 费舍尔变换指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 385 | `FISHER_SIGNAL_10_MKT_Z` | 费舍尔变换指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 386 | `UO_7_14_28_MKT_Z` | 终极震荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 387 | `WT1_10_21_MKT_Z` | Wave Trend 波浪趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 388 | `WT2_10_21_MKT_Z` | Wave Trend 波浪趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 389 | `WT_SPREAD_MKT_Z` | Wave Trend 波浪趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 390 | `CMO_14_MKT_Z` | Chande 动量振荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 391 | `ROC_5_MKT_Z` | 变化率指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 392 | `ROC_10_MKT_Z` | 变化率指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 393 | `ROC_20_MKT_Z` | 变化率指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 394 | `NATR_14_MKT_Z` | 归一化平均真实波幅，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 395 | `GK_VOL_20_MKT_Z` | Garman-Klass 波动率，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 396 | `STD_20_MKT_Z` | 滚动标准差，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 397 | `VHF_28_MKT_Z` | 纵横指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 398 | `VOSC_14_28_MKT_Z` | 成交量振荡指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 399 | `VOL_Z_20_MKT_Z` | 成交量 z-score，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 400 | `POC_GAP_20_MKT_Z` | 收盘价相对近似筹码重心偏离，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 401 | `TURNOVER_SHARE_OF_MARKET_MKT_Z` | 成交额占比指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 402 | `TURNOVER_SHARE_OF_INDUSTRY_MKT_Z` | 成交额占比指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 403 | `ULCER_INDEX_14_MKT_Z` | 溃疡指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 404 | `CHOP_14_MKT_Z` | 趋势噪音指数，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 405 | `MASS_25_MKT_Z` | 梅斯线指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 406 | `SUPER_TREND_DISTANCE_MKT_Z` | SuperTrend 趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 407 | `BBW_20_MKT_Z` | 布林带宽度，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 408 | `PCT_B_20_MKT_Z` | 布林带百分位位置，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 409 | `CCI_20_MKT_Z` | 顺势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 410 | `CCI_TANH_20_MKT_Z` | 顺势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 411 | `INV_WR_14_MKT_Z` | 反向威廉指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 412 | `INV_WR_14_EMA3_MKT_Z` | 反向威廉指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 413 | `CRSI_RSI3_MKT_Z` | ConnorsRSI 分解指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 414 | `CRSI_STREAK_RSI2_MKT_Z` | ConnorsRSI 分解指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 415 | `CRSI_RETURN_PCT_RANK100_MKT_Z` | ConnorsRSI 分解指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 416 | `PVT_REL_20_MKT_Z` | 价量趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 417 | `CLV_MKT_Z` | 收盘位置值，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 418 | `CLV_EMA_5_MKT_Z` | 收盘位置值，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 419 | `PSY_12_MKT_Z` | 心理线指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 420 | `SMR_12_MKT_Z` | 有符号动量比例，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 421 | `AR_LOG_MKT_Z` | ARBR 人气意愿指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 422 | `BR_LOG_MKT_Z` | ARBR 买卖意愿指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 423 | `ARBR_SPREAD_MKT_Z` | ARBR 人气意愿指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 424 | `CR_26_MKT_Z` | CR 能量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 425 | `CR_MA_5_MKT_Z` | CR 能量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 426 | `CR_MA_10_MKT_Z` | CR 能量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 427 | `ANCHOR_VWAP_GAP_MKT_Z` | 锚定 VWAP 指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 428 | `STOCH_RSI_MKT_Z` | 随机 RSI，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 429 | `STOCH_RSI_K_MKT_Z` | 随机 RSI，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 430 | `STOCH_RSI_D_MKT_Z` | 随机 RSI，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 431 | `GMMA_LONG_DISPERSION_MKT_Z` | 顾比复合移动平均线，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 432 | `STC_10_23_50_MKT_Z` | Schaff 趋势周期指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 433 | `SRMI_12_MKT_Z` | 平滑动量指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 434 | `TAPI_MKT_Z` | 成交值价格指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 435 | `TAPI_MA_5_MKT_Z` | 成交值价格指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 436 | `VWAP_DIVERGENCE_MKT_Z` | VWAP 偏离指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 437 | `VWAP_Z_MKT_Z` | VWAP 偏离指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 438 | `VR_26_MKT_Z` | 成交量比率，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 439 | `VSTD_20_MKT_Z` | 成交量标准差，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 440 | `VCI_20_60_MKT_Z` | 波动率压缩指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 441 | `COPPOCK_MKT_Z` | Coppock 曲线，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 442 | `BOTTOM_BUILD_B_5_MKT_Z` | 筑底指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 443 | `MRS_52_MKT_Z` | Mansfield 相对强弱指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 444 | `MRS_40_MKT_Z` | Mansfield 相对强弱指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 445 | `CONDITIONAL_STRENGTH_MKT_Z` | 条件强弱指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 446 | `ASI_MKT_Z` | 累计振动升降指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 447 | `MASS_EFF_MKT_Z` | 梅斯线指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 448 | `REL_RET_INDUSTRY_MKT_Z` | 相对行业收益，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 449 | `MOMENTUM_Z_MKT_Z` | 多周期动量 z-score，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 450 | `TREND_STRENGTH_MKT_Z` | 趋势强度指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 451 | `PP_GAP_MKT_Z` | 枢轴点指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 452 | `R1_GAP_MKT_Z` | R1_GAP，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 453 | `S1_GAP_MKT_Z` | S1_GAP，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 454 | `PVT_DIVERGENCE_20_MKT_Z` | 价量趋势指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 455 | `BOTTOM_BUILD_D_10_MKT_Z` | 筑底指标，经全市场当日截面去极值/标准化后的特征 | 量价技术因子 | 1 | `keep_technical_factor` |
| 456 | `FWD_RET_5D_Z_P01_P99` | 未来 5 个交易日收益率标签，先按 1%/99% 缩尾，再做 z-score 标准化 | 训练标签 | 0 | `keep_label_not_model_input` |
| 457 | `FEATURE_MASK` | 训练样本质量掩码，1 表示该行可用于训练特征 | 训练/交易控制掩码 | 0 | `keep_control_mask_not_model_input` |
| 458 | `BUY_MASK` | 买入可执行掩码，1 表示该日可作为买入候选 | 训练/交易控制掩码 | 0 | `keep_control_mask_not_model_input` |
| 459 | `SELL_MASK` | 卖出可执行掩码，1 表示该日可作为卖出候选 | 训练/交易控制掩码 | 0 | `keep_control_mask_not_model_input` |

## OHLC 派生价格字段对比

### close-relative 口径：以当日后复权收盘价为锚

| 字段 | 原始构造 | 标准化后字段 | 含义 | 使用状态 |
| --- | --- | --- | --- | --- |
| `PV_ADJ_OPEN_TO_CLOSE_RET` | `S_DQ_ADJOPEN / S_DQ_ADJCLOSE - 1` | `PV_ADJ_OPEN_TO_CLOSE_RET_MKT_Z` | 当日开盘价相对当日收盘价的位置，刻画日内从开盘到收盘的方向与幅度 | 463 字段版本新增 |
| `PV_ADJ_HIGH_TO_CLOSE_RET` | `S_DQ_ADJHIGH / S_DQ_ADJCLOSE - 1` | `PV_ADJ_HIGH_TO_CLOSE_RET_MKT_Z` | 当日最高价相对当日收盘价的位置，刻画上影线/盘中冲高回落强度 | 463 字段版本新增 |
| `PV_ADJ_LOW_TO_CLOSE_RET` | `S_DQ_ADJLOW / S_DQ_ADJCLOSE - 1` | `PV_ADJ_LOW_TO_CLOSE_RET_MKT_Z` | 当日最低价相对当日收盘价的位置，刻画下影线/盘中承接强度 | 463 字段版本新增 |
| `S_DQ_ADJCLOSE` | `S_DQ_ADJCLOSE`，实际处理为 `log1p(S_DQ_ADJCLOSE)` 后截面标准化 | `S_DQ_ADJCLOSE_MKT_Z` | 后复权收盘价的截面相对水平，保留价格层级信息 | 463 字段版本新增 |

### preclose-relative 口径：以前一交易日后复权收盘价为锚

| 字段 | 原始构造 | 标准化后字段 | 含义 | 使用状态 |
| --- | --- | --- | --- | --- |
| `PV_ADJ_OPEN_TO_PREVCLOSE_RET` | `S_DQ_ADJOPEN / S_DQ_ADJPRECLOSE - 1` | `PV_ADJ_OPEN_TO_PREVCLOSE_RET_MKT_Z` | 当日开盘相对前收盘的跳空幅度，包含隔夜信息 | 当前字段表已保留 |
| `PV_ADJ_HIGH_TO_PREVCLOSE_RET` | `S_DQ_ADJHIGH / S_DQ_ADJPRECLOSE - 1` | `PV_ADJ_HIGH_TO_PREVCLOSE_RET_MKT_Z` | 当日最高价相对前收盘的涨跌幅，刻画盘中最大上行空间 | 当前字段表已保留 |
| `PV_ADJ_LOW_TO_PREVCLOSE_RET` | `S_DQ_ADJLOW / S_DQ_ADJPRECLOSE - 1` | `PV_ADJ_LOW_TO_PREVCLOSE_RET_MKT_Z` | 当日最低价相对前收盘的涨跌幅，刻画盘中最大下行空间 | 当前字段表已保留 |
| `PV_ADJ_CLOSE_RET_1D` | `S_DQ_ADJCLOSE / S_DQ_ADJPRECLOSE - 1` | `PV_ADJ_CLOSE_RET_1D_MKT_Z` | 当日后复权收盘收益率，等价于 1 日价格动量 | 当前字段表已保留 |

### 口径差异

| 对比项 | close-relative | preclose-relative |
| --- | --- | --- |
| 分母 | 当日后复权收盘价 `S_DQ_ADJCLOSE` | 前一交易日后复权收盘价 `S_DQ_ADJPRECLOSE` |
| 信息类型 | 更偏当日 K 线形态与日内相对位置 | 更偏隔夜跳空、日内极值和单日收益 |
| 是否引入未来函数 | 不引入；只使用 T 日已经形成的 OHLC 数据 | 不引入；使用 T 日 OHLC 与 T-1 收盘 |
| 主要用途 | 给模型补充更“截面形态化”的 OHLC 信息 | 给模型保留传统收益率/动量口径信息 |
| 同时保留的原因 | 与 preclose-relative 不完全等价，能表达收盘相对日内区间的位置 | 与 close-relative 不完全等价，能表达相对昨日收盘的收益路径 |

## 生成日志

```json
{
  "field_trace": "/data/data_process/4.15_revision/model_training_stage1_11_selected_panel/field_decision_trace_all.csv",
  "output": "/data/data_process/4.15_revision/model_training_stage1_11_selected_panel/selected_training_fields_dictionary_cn.md",
  "selected_field_count": 459,
  "model_input_candidate_count": 452,
  "role_counts": {
    "metadata": 3,
    "technical_factor": 104,
    "mask": 196,
    "financial_or_fundamental": 152,
    "label": 1,
    "control_mask": 3
  },
  "forced_exclusions": [
    "APPLICABLE_MISSING_RATE",
    "*_HIGH_MISSING_FLAG"
  ],
  "elapsed_seconds": 0.005
}
```
