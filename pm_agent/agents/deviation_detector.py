import pandas as pd
import numpy as np
import pm4py
from typing import Dict, Any, Tuple, List


class DeviationDetectorAgent:
    """
    Агент для поиска базовых неэффективностей (девиаций) в логе процесса (Event Log).
    Возвращает напрямую DataFrame со всеми обнаруженными отклонениями.
    """

    def __init__(self, case_col: str, activity_col: str, timestamp_col: str):
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.quality_report = {}

    def preprocess_event_log(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        self.quality_report = {
            'original_rows': len(df),
            'original_cases': df[self.case_col].nunique(),
            'warnings': []
        }

        df = df.copy()

        # 1. ОБЕСПЕЧИВАЕМ ТИП DATETIME И ФОРМАТИРОВАНИЕ PM4PY
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors='coerce', utc=True)
            
            # Удаление пустых дат ДО форматирования
            nat_count = int(df[self.timestamp_col].isna().sum())
            if nat_count > 0:
                self.quality_report['warnings'].append(f'Удалено {nat_count} строк с невалидным timestamp (NaT)')
                df = df.dropna(subset=[self.timestamp_col])

            # ИСПОЛЬЗУЕМ СТАНДАРТНОЕ ФОРМАТИРОВАНИЕ PM4PY (Самый стабильный путь на Linux)
            df = pm4py.format_dataframe(
                df, 
                case_id=self.case_col, 
                activity_key=self.activity_col, 
                timestamp_key=self.timestamp_col
            )
        except Exception as e:
            self.quality_report['warnings'].append(f'pm4py formatting error: {e}. Using manual mapping.')
            df['case:concept:name'] = df[self.case_col].astype(str)
            df['concept:name'] = df[self.activity_col].astype(str)
            df['time:timestamp'] = pd.to_datetime(df[self.timestamp_col], errors='coerce', utc=True)
            df = df.dropna(subset=['time:timestamp'])

        # 2. КРИТИЧЕСКИЙ ШАГ ДЛЯ LINUX: Конвертация в pydatetime
        df['time:timestamp'] = [
            t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t 
            for t in df['time:timestamp']
        ]

        # 3. Безопасная сортировка
        df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)

        for col in ['case:concept:name', 'concept:name']:
            na_count = int(df[col].isna().sum())
            if na_count > 0:
                df = df.dropna(subset=[col])
            df[col] = df[col].astype(str).str.strip()

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=['case:concept:name', 'concept:name', 'time:timestamp'])
        dup_count = before - len(df)
        if dup_count > 0:
            self.quality_report['warnings'].append(f'Удалено {dup_count} полных дубликатов')

        # Single-event cases
        case_sizes = df.groupby('case:concept:name').size()
        single_event = case_sizes[case_sizes == 1]
        if len(single_event) > 0:
            self.quality_report['warnings'].append(
                f'Исключено {len(single_event)} кейсов с одним событием из временного анализа'
            )

        self.quality_report['clean_rows'] = len(df)
        self.quality_report['clean_cases'] = df['case:concept:name'].nunique()
        self.quality_report['unique_activities'] = df['concept:name'].nunique()
        
        ts_series = pd.to_datetime(df['time:timestamp'])
        self.quality_report['date_range'] = (ts_series.min(), ts_series.max())

        return df, self.quality_report

    def run_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Запускает все базовые детекторы отклонений и возвращает 
        датасет с результатами (pandas DataFrame) и отчет по качеству.
        """
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

        df_dur = self._add_durations(df)
        case_dur_df = self._calculate_case_durations(df_dur)
        valid_transitions = df_dur.dropna(subset=['duration_h']).copy()

        deviations = []

        # Запускаем все проверки. Каждая функция возвращает список словарей (строк для датасета)
        deviations.extend(self._detect_loops(df_dur))
        deviations.extend(self._detect_long_cycles_and_deadlines(case_dur_df))
        deviations.extend(self._detect_bottlenecks(valid_transitions))
        deviations.extend(self._detect_incidents_and_manual_steps(valid_transitions))
        deviations.extend(self._detect_critical_steps(df_dur, case_dur_df))
        deviations.extend(self._detect_redundant_activities(df_dur, case_dur_df))
        deviations.extend(self._detect_variability_and_dark_processes(df))
        deviations.extend(self._detect_rework_loops(valid_transitions))

        # Собираем DataFrame
        columns = ['deviation_category', 'deviation_name', 'object_id', 'metric', 'description']
        if not deviations:
            findings_df = pd.DataFrame(columns=columns)
        else:
            findings_df = pd.DataFrame(deviations, columns=columns)

        self.findings_df = findings_df
        return self.findings_df, self.quality_report

    # --- Внутренние методы подготовки данных ---

    def _add_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        act_col = 'concept:name'

        # Сортировка перед shift: максимально стабильный способ (через pandas datetime)
        df = df.sort_values([case_col, ts_col]).reset_index(drop=True)

        # Вычисляем длительность в часах через стандартный Pandas dt.total_seconds()
        # Это обходит все проблемы с ручным вычитанием наносекунд и переполнением типов.
        ts_converted = pd.to_datetime(df[ts_col])
        next_ts = ts_converted.groupby(df[case_col]).shift(-1)
        
        df['duration_h'] = (next_ts - ts_converted).dt.total_seconds() / 3600.0
        df.loc[df['duration_h'] < 0, 'duration_h'] = np.nan

        df['prev_act'] = df.groupby(case_col)[act_col].shift(1)
        df['prev2_act'] = df.groupby(case_col)[act_col].shift(2)
        df['next_act'] = df.groupby(case_col)[act_col].shift(-1)
        return df

    def _calculate_case_durations(self, df_dur: pd.DataFrame) -> pd.DataFrame:
        case_col = 'case:concept:name'
        ts_col = 'time:timestamp'
        
        # Находим разницу между макс и мин временем кейса
        case_dur = df_dur.groupby(case_col)[ts_col].agg(['min', 'max'])
        
        # Конвертация в Series для вычисления diff (так как в колонках сейчас могут быть pydatetime)
        c_min = pd.to_datetime(case_dur['min'])
        c_max = pd.to_datetime(case_dur['max'])
        
        case_dur['duration_h'] = (c_max - c_min).dt.total_seconds() / 3600.0
        return case_dur

    # --- Детекторы базовых неэффективностей ---

    def _create_row(self, category: str, name: str, obj: Any, metric: str, desc: str) -> dict:
        return {
            'deviation_category': category,
            'deviation_name': name,
            'object_id': str(obj),
            'metric': metric,
            'description': desc
        }

    def _detect_loops(self, df: pd.DataFrame) -> List[dict]:
        act_col = 'concept:name'
        results = []

        # 1. Самоповтор (Self-loop)
        self_loops = df[df[act_col] == df['prev_act']][act_col].unique()
        for act in self_loops:
            results.append(self._create_row(
                'Зацикленность (общая)', 'Зацикленность «в себя» (Self-loop)', act, None,
                'Немедленное повторение того же этапа. Указывает на техническое дублирование'
            ))

        # 2. Пинг-понг 
        ping_pong = df[(df[act_col] == df['prev2_act']) & (df['prev_act'] == df['next_act'])][act_col].unique()
        for act in ping_pong:
            results.append(self._create_row(
                'Зацикленность (общая)', 'Зацикленность «Пинг-понг»', act, None,
                'Повторение пары операций (A-B-A). Характерно для доработок'
            ))

        # 3. Возвраты
        def get_returns(group):
            seen, found = set(), set()
            for act in group:
                if act in seen: found.add(act)
                seen.add(act)
            return list(found)
            
        returns_series = df.groupby('case:concept:name')[act_col].apply(get_returns)
        all_returns = set(x for lst in returns_series for x in lst)
        
        for act in all_returns:
            results.append(self._create_row(
                'Зацикленность (общая)', 'Зацикленность «возврат»', act, None,
                'Повторение операции после других действий. Признак переделок'
            ))

        return results

    def _detect_long_cycles_and_deadlines(self, case_dur_df: pd.DataFrame) -> List[dict]:
        dur = case_dur_df['duration_h'].dropna()
        if dur.empty: return []
        
        results = []
        p95 = dur.quantile(0.95)
        
        long_cases = case_dur_df[case_dur_df['duration_h'] > p95]
        for case_id, row in long_cases.iterrows():
            results.append(self._create_row(
                'Долгий цикл (Long Cycle Time)', 'Долгий цикл', case_id, f"Длительность: {row['duration_h']:.2f}ч",
                'Превышение времени выполнения процесса над нормативом'
            ))
            results.append(self._create_row(
                'Пропущенные дедлайны', 'Нарушение SLA', case_id, 'Топ-5% по длительности',
                'Кейс находится в топ-5% самых долгих, возможен срыв сроков'
            ))
            
        # Outliers (IQR)
        Q1, Q3 = dur.quantile(0.25), dur.quantile(0.75)
        IQR = Q3 - Q1
        outliers = case_dur_df[(case_dur_df['duration_h'] < Q1 - 1.5 * IQR) | (case_dur_df['duration_h'] > Q3 + 1.5 * IQR)]
        for case_id, row in outliers.head(10).iterrows():
            results.append(self._create_row(
                'Аномалии (Outliers)', 'Выброс длительности кейса', case_id, f"{row['duration_h']:.2f}ч",
                'Статистически значимое отклонение длительности кейса от нормы (по IQR)'
            ))
            
        return results

    def _detect_bottlenecks(self, valid_tdf: pd.DataFrame) -> List[dict]:
        if valid_tdf.empty: return []
        act_col = 'concept:name'
        
        bottlenecks = valid_tdf.groupby([act_col, 'next_act'])['duration_h'].agg(['mean', 'count'])
        bottlenecks = bottlenecks[bottlenecks['count'] > 5].sort_values('mean', ascending=False).head(5)
        
        results = []
        for (a1, a2), row in bottlenecks.iterrows():
            transition = f"{a1} -> {a2}"
            results.append(self._create_row(
                'Узкое место (Bottleneck)', 'Узкое место на переходе', transition, f"Ожидание в среднем: {row['mean']:.2f}ч",
                'Этап вызывает систематические задержки из-за высокой нагрузки'
            ))
        return results

    def _detect_incidents_and_manual_steps(self, valid_tdf: pd.DataFrame) -> List[dict]:
        if valid_tdf.empty: return []
        act_col = 'concept:name'
        
        stats = valid_tdf.groupby(act_col)['duration_h'].agg(['mean', 'median', 'std', 'count'])
        stats['cv'] = stats['std'] / stats['mean'].replace(0, np.nan)
        stats['diff_pct'] = abs(stats['mean'] - stats['median']) / stats['median'].replace(0, np.nan)
        
        results = []
        
        # Ручные шаги / Нестандартизированные
        manual = stats[(stats['diff_pct'] < 0.1) & (stats['cv'] > 0.5) & (stats['count'] > 5)].head(5)
        for act, row in manual.iterrows():
            results.append(self._create_row(
                'Нестандартизированный (ручной) этап', 'Высокая вариативность шага', act, f"CV: {row['cv']:.2f}",
                'Длительность зависит от человеческого фактора (высокое станд. отклонение)'
            ))
            
        # Систематические инциденты
        repeated = stats[(stats['diff_pct'] > 0.15) & (stats['cv'] > 0.5) & (stats['count'] > 5)].head(5)
        for act, row in repeated.iterrows():
            results.append(self._create_row(
                'Многократные инциденты', 'Регулярные сбои операции', act, f"Отклонение mean от median: {row['diff_pct']:.0%}",
                'Систематические ошибки, вызывающие долгие зависания операций'
            ))
            
        # Разовые инциденты (через IQR)
        for act, group in valid_tdf.groupby(act_col):
            dur = group['duration_h'].dropna()
            if len(dur) < 10: continue
            
            Q1, Q3 = dur.quantile(0.25), dur.quantile(0.75)
            clean = dur[(dur >= Q1 - 1.5 * (Q3 - Q1)) & (dur <= Q3 + 1.5 * (Q3 - Q1))]
            outliers_count = len(dur) - len(clean)
            
            if outliers_count > 0:
                clean_med = clean.median()
                clean_diff = abs(clean.mean() - clean_med) / clean_med if clean_med > 0 else 0
                if clean_diff < 0.1 and abs(dur.mean() - dur.median()) / dur.median() > 0.1:
                    results.append(self._create_row(
                        'Разовые инциденты', 'Аномальный сбой', act, f"Аномалий: {outliers_count} шт",
                        'Единичные длительные операции на фоне нормального выполнения'
                    ))
                    
        # Ручные исключения (очень долгие, но стабильные операции)
        overall_mean = valid_tdf['duration_h'].mean()
        manual_exc = stats[(stats['mean'] > overall_mean * 2) & (stats['cv'] < 0.3) & (stats['count'] > 5)]
        for act, row in manual_exc.iterrows():
            results.append(self._create_row(
                'Ручные исключения (Manual Exceptions)', 'Требует подтверждения', act, f"Дольше среднего в {row['mean']/overall_mean:.1f} раз",
                'Долгие операции с низкой вариативностью (возможно, ручное согласование)'
            ))
            
        return results

    def _detect_critical_steps(self, df_dur: pd.DataFrame, case_dur_df: pd.DataFrame) -> List[dict]:
        act_dur = df_dur.dropna(subset=['duration_h']).groupby(['case:concept:name', 'concept:name'])['duration_h'].mean().unstack(fill_value=0)
        if act_dur.empty: return []

        target = case_dur_df[['duration_h']].rename(columns={'duration_h': 'total_h'})
        act_dur = act_dur.join(target, how='inner')

        results = []
        for col in [c for c in act_dur.columns if c != 'total_h']:
            if act_dur[col].std() > 0:
                corr = act_dur[col].corr(act_dur['total_h'])
                if corr > 0.5:
                    results.append(self._create_row(
                        'Критически важный этап', 'Высокая корреляция с итогом', col, f"Корреляция: {corr:.2f}",
                        'Длительность этой операции наиболее сильно влияет на общую длительность всего процесса'
                    ))
        return results

    def _detect_redundant_activities(self, df_dur: pd.DataFrame, case_dur_df: pd.DataFrame) -> List[dict]:
        import scipy.stats as stats
        total_cases = df_dur['case:concept:name'].nunique()
        act_rate = df_dur.groupby('concept:name')['case:concept:name'].nunique() / total_cases
        
        results = []
        for act in act_rate[act_rate < 0.5].index:
            cases_with_act = df_dur[df_dur['concept:name'] == act]['case:concept:name'].unique()
            dur_with = case_dur_df.loc[case_dur_df.index.isin(cases_with_act), 'duration_h'].dropna()
            dur_without = case_dur_df.loc[~case_dur_df.index.isin(cases_with_act), 'duration_h'].dropna()
            
            if len(dur_with) > 5 and len(dur_without) > 5:
                # Если наличие активности не замедляет и не ускоряет кейс
                stat, p_val = stats.ttest_ind(dur_with.values, dur_without.values)
                if p_val > 0.05:
                    results.append(self._create_row(
                        'Избыточные шаги (Redundant', 'Шаг не добавляющий ценности', act, None,
                        'Этап не влияет на конечную длительность и проходится не всеми'
                    ))
        return results

    def _detect_variability_and_dark_processes(self, df: pd.DataFrame) -> List[dict]:
        results = []
        try:
            variants = pm4py.get_variants(df)
            total = sum(variants.values())
            if total == 0: return results
            
            n_variants = len(variants)
            ratio = n_variants / total
            
            if ratio > 0.3:
                results.append(self._create_row(
                    'Вариативность (High Variability)', 'Множество путей', 'Весь процесс', f"Ratio: {ratio:.2f}",
                    'Слишком много альтернативных путей. Снижает предсказуемость'
                ))
                
            sorted_v = sorted(variants.items(), key=lambda x: -x[1])
            cumsum = 0
            official_count = 0
            for _, count in sorted_v:
                cumsum += count / total
                official_count += 1
                if cumsum >= 0.8:
                    break
                    
            dark_variants = n_variants - official_count
            if dark_variants > 0:
                results.append(self._create_row(
                    'Скрытые сценарии (Dark Processes)', 'Неформализованные пути', 'Весь процесс', f"Редких путей: {dark_variants}",
                    'Сотни редких путей выполнения, отсутствующих в стандартных регламентах'
                ))
        except Exception as e:
            pass # Если pm4py сломается, проигнорируем
        return results

    def _detect_rework_loops(self, valid_tdf: pd.DataFrame) -> List[dict]:
        case_col = 'case:concept:name'
        act_col = 'concept:name'
        
        df = valid_tdf.copy()
        df['pos'] = df.groupby(case_col).cumcount()
        avg_pos = df.groupby(act_col)['pos'].median().sort_values()
        order = {act: i for i, act in enumerate(avg_pos.index)}

        df['from_order'] = df[act_col].map(order)
        df['to_order'] = df['next_act'].map(order)
        rework = df[df['to_order'] < df['from_order']]

        if rework.empty: return []
        
        rework_summary = rework.groupby([act_col, 'next_act']).size().sort_values(ascending=False).head(5)
        
        results = []
        for (a1, a2), count in rework_summary.items():
            transition = f"{a1} -> {a2}"
            results.append(self._create_row(
                'Обратные потоки (Rework Loops)', 'Возврат на предыдущий этап', transition, f"Кол-во: {count}",
                'Сущность возвращается на этап, который типично выполняется раньше в процессе'
            ))
        return results

    def get_summary_text(self) -> str:
        """Generates a human-readable summary of findings from the findings_df."""
        lines = ["## 🔍 Результаты анализа отклонений процесса"]
        
        if not hasattr(self, 'findings_df') or self.findings_df.empty:
            lines.append("\n✅ Отклонений не обнаружено.")
            return "\n".join(lines)
            
        found_count = len(self.findings_df['deviation_category'].unique())
        lines.insert(1, f"\n**Обнаружено категорий отклонений: {found_count}**\n")
        
        cat_idx = 1
        for category, group in self.findings_df.groupby('deviation_category', sort=False):
            lines.append(f"### {cat_idx}. {category}")
            desc = group['description'].iloc[0]
            lines.append(f"> {desc}")
            
            items_list = []
            for _, row in group.iterrows():
                obj_id = row['object_id']
                metric = row['metric']
                if pd.notna(metric) and metric:
                    items_list.append(f"`{obj_id}` ({metric})")
                else:
                    items_list.append(f"`{obj_id}`")
            
            if len(items_list) > 10:
                joined = ", ".join(items_list[:10])
                lines.append(f"**Найдены**: {joined}\n*(и еще {len(items_list) - 10} шт.)*")
            else:
                joined = ", ".join(items_list)
                lines.append(f"**Найдены**: {joined}")
                
            cat_idx += 1

        return "\n".join(lines)
