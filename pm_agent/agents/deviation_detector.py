import pandas as pd
import numpy as np
import pm4py
from scipy import stats
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime


class DeviationDetectorAgent:
    def __init__(self, case_col: str, activity_col: str, timestamp_col: str):
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.quality_report = {}
        self.findings = {}

    def preprocess_event_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robust preprocessing of event log.
        """
        self.quality_report = {
            'original_rows': len(df),
            'original_cases': df[self.case_col].nunique(),
            'warnings': []
        }

        df = df.copy()

        # Parse timestamp
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], infer_datetime_format=True, errors='coerce')
        nat_count = df[self.timestamp_col].isna().sum()
        if nat_count > 0:
            self.quality_report['warnings'].append(f'Удалено {nat_count} строк с невалидным timestamp (NaT)')
            df = df.dropna(subset=[self.timestamp_col])

        # Null/NaN in case_id and activity
        for col, name in [(self.case_col, 'case_id'), (self.activity_col, 'activity')]:
            na_count = df[col].isna().sum()
            if na_count > 0:
                self.quality_report['warnings'].append(f'Удалено {na_count} строк с пустым {name}')
                df = df.dropna(subset=[col])
            df[col] = df[col].astype(str)

        # Clean activity names
        df[self.activity_col] = df[self.activity_col].str.strip()

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=[self.case_col, self.activity_col, self.timestamp_col])
        dup_count = before - len(df)
        if dup_count > 0:
            self.quality_report['warnings'].append(f'Удалено {dup_count} полных дубликатов (case+activity+timestamp)')

        # Sort (using sequential stable sort to avoid pandas lexsort AssertionError on DatetimeArray)
        df = df.sort_values(self.timestamp_col)
        df = df.sort_values(self.case_col, kind='stable').reset_index(drop=True)

        # Single-event cases
        case_sizes = df.groupby(self.case_col).size()
        single_event = case_sizes[case_sizes == 1]
        if len(single_event) > 0:
            self.quality_report['warnings'].append(
                f'Обнаружено {len(single_event)} кейсов с одним событием '
                f'({len(single_event)/len(case_sizes)*100:.1f}%) — они исключены из временного анализа'
            )
            self.quality_report['single_event_cases_count'] = len(single_event)

        # Formatting for pm4py (bypassing buggy pm4py.format_dataframe for pandas datetimes)
        df['case:concept:name'] = df[self.case_col]
        df['concept:name'] = df[self.activity_col]
        df['time:timestamp'] = df[self.timestamp_col]

        # Stats
        self.quality_report['clean_rows'] = len(df)
        self.quality_report['clean_cases'] = df['case:concept:name'].nunique()
        self.quality_report['unique_activities'] = df['concept:name'].nunique()
        self.quality_report['date_range'] = (df['time:timestamp'].min(), df['time:timestamp'].max())
        
        return df, self.quality_report

    def add_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds durations and context (prev/next) for analyzers."""
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        act_col = 'concept:name'
        
        df = df.sort_values([case_col, ts_col]).copy()
        df['next_ts'] = df.groupby(case_col)[ts_col].shift(-1)
        df['duration_h'] = (df['next_ts'] - df[ts_col]).dt.total_seconds() / 3600

        # Protect against negative durations
        neg = df['duration_h'] < 0
        if neg.any():
            df.loc[neg, 'duration_h'] = np.nan

        df['prev_act'] = df.groupby(case_col)[act_col].shift(1)
        df['next_act'] = df.groupby(case_col)[act_col].shift(-1)
        return df

    def run_analysis(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Runs all 18 deviation detectors. Expects preprocessed DataFrame."""
        df_dur = self.add_durations(df)
        
        # Mapping analyzers to findings
        self.findings = {
            'outliers_70': self._detect_duration_outliers(df_dur),
            'loops_71_75': self._detect_all_loops(df_dur),
            'long_cycles_76': self._detect_long_cycles(df_dur),
            'bottlenecks_77': self._detect_bottlenecks(df_dur),
            'manual_steps_78': self._detect_manual_steps(df_dur),
            'one_time_incidents_79': self._detect_one_time_incidents(df_dur),
            'repeated_incidents_80': self._detect_repeated_incidents(df_dur),
            'missed_deadlines_81': self._detect_missed_deadlines(df_dur),
            'critical_steps_82': self._detect_critical_steps(df_dur),
            'redundant_activities_83': self._detect_redundant_activities(df_dur),
            'variability_84': self._detect_high_variability(df_dur),
            'dark_processes_85': self._detect_dark_processes(df_dur),
            'manual_exceptions_86': self._detect_manual_exceptions(df_dur),
            'rework_loops_87': self._detect_rework_loops(df_dur)
        }
        
        return self.findings, self.quality_report

    # --- Individual Detectors Implementation ---

    def _detect_duration_outliers(self, df):
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        case_dur = df.groupby(case_col)[ts_col].agg(['min', 'max'])
        case_dur['duration'] = (case_dur['max'] - case_dur['min']).dt.total_seconds() / 3600
        if case_dur.empty: return []
        Q1, Q3 = case_dur['duration'].quantile(0.25), case_dur['duration'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = case_dur[(case_dur['duration'] < Q1 - 1.5 * IQR) | (case_dur['duration'] > Q3 + 1.5 * IQR)]
        return outliers.index.tolist()[:10]  # Return top 10 case IDs

    def _detect_all_loops(self, df):
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        df = df.sort_values([case_col, 'time:timestamp'])
        df['prev_act'] = df.groupby(case_col)[act_col].shift(1)
        df['prev2_act'] = df.groupby(case_col)[act_col].shift(2)
        df['next_act'] = df.groupby(case_col)[act_col].shift(-1)
        
        results = {
            'self_loops': df[df[act_col] == df['prev_act']][act_col].unique().tolist(),
            'ping_pong': df[(df[act_col] == df['prev2_act']) & (df['prev_act'] == df['next_act'])][act_col].unique().tolist()
        }
        
        # Returns
        def has_return_val(group):
            seen = set()
            found = []
            for act in group[act_col]:
                if act in seen: found.append(act)
                seen.add(act)
            return list(set(found))
        
        returns = df.groupby(case_col).apply(has_return_val)
        all_returns = set()
        for r in returns: all_returns.update(r)
        results['returns'] = list(all_returns)
        
        return results

    def _detect_long_cycles(self, df):
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        case_dur = df.groupby(case_col)[ts_col].agg(['min', 'max'])
        case_dur['duration_h'] = (case_dur['max'] - case_dur['min']).dt.total_seconds() / 3600
        if case_dur.empty: return []
        threshold = case_dur['duration_h'].quantile(0.95)
        long_cases = case_dur[case_dur['duration_h'] > threshold]
        return {'threshold_h': round(threshold, 2), 'cases': long_cases.index.tolist()[:5]}

    def _detect_bottlenecks(self, df):
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        ts_col = 'time:timestamp'
        transition_stats = df.dropna(subset=['duration_h']).copy()
        if transition_stats.empty: return []
        
        bottlenecks = transition_stats.groupby([act_col, 'next_act'])['duration_h'].agg(['mean', 'count'])
        bottlenecks = bottlenecks[bottlenecks['count'] > 5].sort_values('mean', ascending=False)
        return bottlenecks.head(5).to_dict('index')

    def _detect_manual_steps(self, df):
        act_col = 'concept:name'
        df_valid = df.dropna(subset=['duration_h'])
        if df_valid.empty: return []
        stats = df_valid.groupby(act_col)['duration_h'].agg(['mean', 'median', 'std', 'count'])
        stats['cv'] = stats['std'] / stats['mean']
        stats['mean_median_diff'] = abs(stats['mean'] - stats['median']) / stats['median'].replace(0, np.nan)
        manual = stats[(stats['mean_median_diff'] < 0.1) & (stats['cv'] > 0.5) & (stats['count'] > 5)]
        return manual.head(5).to_dict('index')

    def _detect_one_time_incidents(self, df):
        act_col = 'concept:name'
        df_valid = df.dropna(subset=['duration_h'])
        results = []
        for act, group in df_valid.groupby(act_col):
            dur = group['duration_h']
            if len(dur) < 10: continue
            mean_val, med_val = dur.mean(), dur.median()
            diff_pct = abs(mean_val - med_val) / med_val if med_val > 0 else 0
            if diff_pct > 0.1:
                Q1, Q3 = dur.quantile(0.25), dur.quantile(0.75)
                IQR = Q3 - Q1
                clean = dur[(dur >= Q1 - 1.5 * IQR) & (dur <= Q3 + 1.5 * IQR)]
                if not clean.empty:
                    clean_diff = abs(clean.mean() - clean.median()) / clean.median() if clean.median() > 0 else 0
                    if clean_diff < 0.1:
                        results.append({'activity': act, 'outliers': len(dur) - len(clean)})
        return results

    def _detect_repeated_incidents(self, df):
        act_col = 'concept:name'
        df_valid = df.dropna(subset=['duration_h'])
        if df_valid.empty: return []
        stats = df_valid.groupby(act_col)['duration_h'].agg(['mean', 'median', 'std', 'count'])
        stats['cv'] = stats['std'] / stats['mean']
        stats['diff_pct'] = abs(stats['mean'] - stats['median']) / stats['median'].replace(0, np.nan)
        repeated = stats[(stats['diff_pct'] > 0.15) & (stats['cv'] > 0.5) & (stats['count'] > 5)]
        return repeated.head(5).to_dict('index')

    def _detect_missed_deadlines(self, df):
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        case_dur = df.groupby(case_col)[ts_col].agg(['min', 'max'])
        case_dur['h'] = (case_dur['max'] - case_dur['min']).dt.total_seconds() / 3600
        if case_dur.empty: return []
        p95 = case_dur['h'].quantile(0.95)
        return case_dur[case_dur['h'] > p95].index.tolist()[:5]

    def _detect_critical_steps(self, df):
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        ts_col = 'time:timestamp'
        case_dur = df.groupby(case_col)[ts_col].agg(['min', 'max'])
        case_dur['total_h'] = (case_dur['max'] - case_dur['min']).dt.total_seconds() / 3600
        
        act_dur = df.dropna(subset=['duration_h']).groupby([case_col, act_col])['duration_h'].mean().unstack(fill_value=0)
        if act_dur.empty: return []
        act_dur = act_dur.join(case_dur['total_h'])
        
        correlations = {}
        for col in act_dur.columns:
            if col != 'total_h' and act_dur[col].std() > 0:
                corr = act_dur[col].corr(act_dur['total_h'])
                if corr > 0.5: correlations[col] = round(corr, 2)
        return correlations

    def _detect_redundant_activities(self, df):
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        ts_col = 'time:timestamp'
        total_cases = df[case_col].nunique()
        act_presence = df.groupby(act_col)[case_col].nunique()
        act_rate = act_presence / total_cases
        
        case_dur = df.groupby(case_col)[ts_col].agg(['min', 'max'])
        case_dur['duration_h'] = (case_dur['max'] - case_dur['min']).dt.total_seconds() / 3600
        
        redundant = []
        for act in act_rate[act_rate < 0.5].index:
            cases_with = df[df[act_col] == act][case_col].unique()
            dur_with = case_dur.loc[case_dur.index.isin(cases_with), 'duration_h']
            dur_without = case_dur.loc[~case_dur.index.isin(cases_with), 'duration_h']
            if len(dur_with) > 5 and len(dur_without) > 5:
                _, p_val = stats.ttest_ind(dur_with, dur_without)
                if p_val > 0.05: redundant.append(act)
        return redundant

    def _detect_high_variability(self, df):
        variants = pm4py.get_variants(df)
        total_cases = sum(variants.values())
        if total_cases == 0: return {}
        n_variants = len(variants)
        return {
            'variant_count': n_variants,
            'ratio': round(n_variants / total_cases, 2),
            'high': (n_variants / total_cases) > 0.3
        }

    def _detect_dark_processes(self, df):
        variants = pm4py.get_variants(df)
        total = sum(variants.values())
        if total == 0: return {}
        sorted_v = sorted(variants.items(), key=lambda x: -x[1])
        cumsum = 0
        official_count = 0
        for _, count in sorted_v:
            cumsum += count / total
            official_count += 1
            if cumsum >= 0.8: break
        
        dark_variants = len(variants) - official_count
        return {'dark_variant_count': dark_variants, 'official_coverage': round(cumsum, 2)}

    def _detect_manual_exceptions(self, df):
        act_col = 'concept:name'
        df_valid = df.dropna(subset=['duration_h'])
        if df_valid.empty: return []
        stats_df = df_valid.groupby(act_col)['duration_h'].agg(['mean', 'std', 'count'])
        stats_df['cv'] = stats_df['std'] / stats_df['mean']
        overall_mean = df_valid['duration_h'].mean()
        manual_exc = stats_df[(stats_df['mean'] > overall_mean * 2) & (stats_df['cv'] < 0.3) & (stats_df['count'] > 5)]
        return manual_exc.index.tolist()

    def _detect_rework_loops(self, df):
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        df = df.sort_values([case_col, 'time:timestamp'])
        df['pos'] = df.groupby(case_col).cumcount()
        avg_pos = df.groupby(act_col)['pos'].median().sort_values()
        order = {act: i for i, act in enumerate(avg_pos.index)}
        
        df['next_act'] = df.groupby(case_col)[act_col].shift(-1)
        df_trans = df.dropna(subset=['next_act']).copy()
        if df_trans.empty: return []
        df_trans['from_order'] = df_trans[act_col].map(order)
        df_trans['to_order'] = df_trans['next_act'].map(order)
        rework = df_trans[df_trans['to_order'] < df_trans['from_order']]
        
        rework_summary = rework.groupby([act_col, 'next_act']).size().sort_values(ascending=False)
        return rework_summary.head(5).to_dict()

    # --- Result Formatting for Report ---
    
    def get_summary_text(self) -> str:
        """Generates a human-readable summary of findings. Only found deviations are shown."""
        lines = ["## 🔍 Результаты анализа отклонений процесса"]
        found_count = 0

        # 1. Outliers
        outliers = self.findings.get('outliers_70', [])
        if outliers:
            found_count += 1
            lines.append(f"### {found_count}. Аномалии (Outliers)")
            lines.append("> Редкие или статистически значимые события, которые существенно отклоняются от нормы.")
            lines.append(f"**Найдены**: {', '.join(outliers)}")
            lines.append(f"*Где искать:* Проверьте кейсы {', '.join(outliers[:5])} на предмет ошибок ввода данных или уникальных сбоев.")

        # 2-6. Loops
        loops = self.findings.get('loops_71_75', {})
        if any(loops.values()):
            found_count += 1
            lines.append(f"### {found_count}. Зацикленность (общая)")
            lines.append("> Любые случаи повторного возникновения операции в рамках одного кейса.")
            lines.append("**Обнаружена**")

        if loops.get('self_loops'):
            found_count += 1
            lines.append(f"### {found_count}. Зацикленность «в себя» (Self-loop)")
            lines.append("> Немедленное повторение того же этапа. Часто указывает на техническое дублирование записей.")
            lines.append(f"**Найдены**: {', '.join(loops['self_loops'])}")

        if loops.get('returns'):
            found_count += 1
            lines.append(f"### {found_count}. Зацикленность «возврат»")
            lines.append("> Повторение операции после выполнения других действий. Признак переделок.")
            lines.append(f"**Найдены**: {', '.join(loops['returns'])}")

        if loops.get('ping_pong'):
            found_count += 1
            lines.append(f"### {found_count}. Зацикленность «Пинг-понг»")
            lines.append("> Повторение пары операций (A-B-A). Характерно для доработок или пересылки между отделами.")
            lines.append(f"**Найдены**: {', '.join(loops['ping_pong'])}")

        if loops.get('back_to_start'):
            found_count += 1
            lines.append(f"### {found_count}. Зацикленность «В начало»")
            lines.append("> Возврат из середины процесса на самый первый этап. Признак полной отмены и перезапуска.")
            lines.append("**Обнаружена**")

        # 7. Long Cycles
        lc = self.findings.get('long_cycles_76', {})
        if lc.get('cases'):
            found_count += 1
            thr = lc.get('threshold_h')
            lines.append(f"### {found_count}. Долгий цикл (Long Cycle Time)")
            lines.append("> Превышение времени выполнения процесса над нормативными или типичными значениями.")
            lines.append(f"**Обнаружен**: кейсы дольше {thr}ч")

        # 8. Bottlenecks
        bn = self.findings.get('bottlenecks_77', [])
        if bn:
            found_count += 1
            lines.append(f"### {found_count}. Узкое место (Bottleneck)")
            lines.append("> Этапы, вызывающие систематические задержки из-за высокой нагрузки или нехватки ресурсов.")
            lines.append("**Топ задержек по переходам**:")
            for k, v in bn.items():
                lines.append(f"  - `{k[0]} -> {k[1]}` (ожидание в среднем: {v['mean']:.2f}ч)")

        # 9. Manual Steps
        ms = self.findings.get('manual_steps_78', [])
        if ms:
            found_count += 1
            lines.append(f"### {found_count}. Нестандартизированный (ручной) этап")
            lines.append("> Этапы с высокой вариативностью длительности, зависящие от человеческого фактора.")
            lines.append(f"**Найдены**: {', '.join(ms.keys())}")

        # 10. One-time Incidents
        oti = self.findings.get('one_time_incidents_79', [])
        if oti:
            found_count += 1
            lines.append(f"### {found_count}. Разовые инциденты")
            lines.append("> Длительные операции, вызванные редким сбоем или аварией (не систематические).")
            lines.append(f"**Найдены**: {', '.join([d['activity'] for d in oti[:10]])}")
            if len(oti) > 10: lines.append(f"*(и еще {len(oti)-10} этапов)*")

        # 11. Repeated Incidents
        ri = self.findings.get('repeated_incidents_80', [])
        if ri:
            found_count += 1
            lines.append(f"### {found_count}. Многократные инциденты")
            lines.append("> Систематические ошибки или программные сбои, вызывающие долгие операции.")
            lines.append(f"**Найдены**: {', '.join(ri.keys())}")

        # 12. Missed Deadlines
        md = self.findings.get('missed_deadlines_81', [])
        if md:
            found_count += 1
            lines.append(f"### {found_count}. Пропущенные дедлайны")
            lines.append("> Нарушение временных ограничений процесса (кейсы в топ-5% по длительности).")
            lines.append(f"**Найдены**: {', '.join(md)}")

        # 13. Critical Steps
        cs = self.findings.get('critical_steps_82', [])
        if cs:
            found_count += 1
            lines.append(f"### {found_count}. Критически важный этап")
            lines.append("> Операции, длительность которых наиболее сильно влияет на общую длительность всего процесса.")
            lines.append(f"**Найдены**: {', '.join(cs.keys())}")
            lines.append(f"*Совет:* Оптимизация именно этих этапов даст максимальный эффект для ускорения процесса.")

        # 14. Redundant Activities
        ra = self.findings.get('redundant_activities_83', [])
        if ra:
            found_count += 1
            lines.append(f"### {found_count}. Избыточные шаги (Redundant Activities)")
            lines.append("> Действия, не добавляющие ценности и не влияющие на конечный результат.")
            lines.append(f"**Найдены**: {', '.join(ra)}")

        # 15. Variability
        hv = self.findings.get('variability_84', {})
        if hv.get('high'):
            found_count += 1
            lines.append(f"### {found_count}. Вариативность (High Variability)")
            lines.append("> Слишком много альтернативных путей выполнения. Снижает предсказуемость процесса.")
            lines.append(f"**Статус**: ⚠️ Высокая вариативность ({hv.get('variant_count')} путей, Ratio: {hv.get('ratio')})")

        # 16. Dark Processes
        dp = self.findings.get('dark_processes_85', {})
        if dp.get('dark_variant_count'):
            found_count += 1
            lines.append(f"### {found_count}. Скрытые сценарии (Dark Processes)")
            lines.append("> Неформализованные пути выполнения, отсутствующие в стандартных регламентах.")
            lines.append(f"**Обнаружено**: {dp.get('dark_variant_count')} редких путей (покрывают {(1-dp.get('official_coverage'))*100:.1f}% трафика)")

        # 17. Manual Exceptions
        me = self.findings.get('manual_exceptions_86', [])
        if me:
            found_count += 1
            lines.append(f"### {found_count}. Ручные исключения (Manual Exceptions)")
            lines.append("> Долгие операции с низкой вариативностью, требующие ручного подтверждения.")
            lines.append(f"**Найдены**: {', '.join(me)}")

        # 18. Rework Loops
        rw = self.findings.get('rework_loops_87', [])
        if rw:
            found_count += 1
            lines.append(f"### {found_count}. Обратные потоки (Rework Loops)")
            lines.append("> Непредусмотренные возвраты на предыдущие этапы из-за ошибок или доработок.")
            lines.append("**Топ переходов-возвратов**:")
            for k, v in rw.items():
                lines.append(f"  - `{k[0]} -> {k[1]}` (повторено {v} раз)")

        if found_count == 0:
            lines.append("\n✅ Отклонений не обнаружено.")
        else:
            lines.insert(1, f"\n**Обнаружено отклонений: {found_count}**\n")

        return "\n".join(lines)
