import logging
from itertools import permutations
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from traincheck.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    GroupedPreconditions,
    Hypothesis,
    Invariant,
    Relation,
)
from traincheck.invariant.lead_relation import (
    check_same_level,
    get_func_data_per_PT,
    get_func_names_to_deal_with,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.trace.trace_pandas import TracePandas


# 常量定义
class Constants:
    EXP_GROUP_NAME = "func_cover"
    FUNCTION_CALL_PRE = "function_call (pre)"
    EBPF_FILTER = "[eBPF]"  # 新增：eBPF过滤标识


def _filter_ebpf_functions(function_pool: Set[Any]) -> Set[Any]:
    """过滤出包含[eBPF]的函数"""
    return {func for func in function_pool if Constants.EBPF_FILTER in str(func)}


def _is_ebpf_function(func_name: str) -> bool:
    """检查函数名是否包含[eBPF]"""
    return Constants.EBPF_FILTER in func_name


class TraceCache:
    """Trace数据的缓存管理器"""

    def __init__(self, trace: TracePandas):
        self.trace = trace
        self.logger = logging.getLogger(__name__)

    def get_or_compute_function_pool(self) -> Set[Any]:
        """获取或计算函数池（只包含eBPF函数）"""
        if self.trace.function_pool is not None:
            # 如果已经缓存，需要检查是否已经过滤过eBPF函数
            if hasattr(self.trace, "_ebpf_filtered") and self.trace._ebpf_filtered:
                return self.trace.function_pool
            else:
                # 重新过滤
                filtered_pool = _filter_ebpf_functions(self.trace.function_pool)
                self.trace.function_pool = filtered_pool
                self.trace._ebpf_filtered = True
                return filtered_pool

        function_pool = set(get_func_names_to_deal_with(self.trace))
        # 过滤出包含[eBPF]的函数
        filtered_pool = _filter_ebpf_functions(function_pool)

        self.trace.function_pool = filtered_pool
        self.trace._ebpf_filtered = True

        if len(filtered_pool) == 0:
            self.logger.warning("No eBPF functions found in the trace")
        else:
            self.logger.info(f"Found {len(filtered_pool)} eBPF functions")

        return filtered_pool

    def get_or_compute_function_data(self) -> Tuple[Dict, Dict, Dict]:
        """获取或计算函数数据"""
        if all(
            [
                self.trace.function_times is not None,
                self.trace.function_id_map is not None,
                self.trace.listed_events is not None,
            ]
        ):
            return (
                self.trace.function_times,
                self.trace.function_id_map,
                self.trace.listed_events,
            )

        function_pool = self.get_or_compute_function_pool()
        if len(function_pool) == 0:
            self.logger.warning("No relevant eBPF function calls found")
            return {}, {}, {}

        print("Start preprocessing (eBPF functions only)....")
        function_times, function_id_map, listed_events = get_func_data_per_PT(
            self.trace, function_pool
        )

        # 缓存结果
        self.trace.function_times = function_times
        self.trace.function_id_map = function_id_map
        self.trace.listed_events = listed_events
        print("End preprocessing")

        return function_times, function_id_map, listed_events

    def get_or_compute_relations(self) -> Tuple[Dict, Dict]:
        """获取或计算同级关系（只考虑eBPF函数）"""
        if all(
            [
                self.trace.same_level_func_cover is not None,
                self.trace.valid_relations_cover is not None,
            ]
        ):
            # 检查是否已经是eBPF过滤后的结果
            if (
                hasattr(self.trace, "_ebpf_relations_filtered")
                and self.trace._ebpf_relations_filtered
            ):
                return (
                    self.trace.same_level_func_cover,
                    self.trace.valid_relations_cover,
                )

        function_times, function_id_map, listed_events = (
            self.get_or_compute_function_data()
        )
        function_pool = self.get_or_compute_function_pool()

        print("Start same level checking (eBPF functions only)...")
        same_level_func = {}
        valid_relations = {}

        for (process_id, thread_id), _ in tqdm(
            listed_events.items(), ascii=True, leave=True, desc="Groups Processed"
        ):
            same_level_func[(process_id, thread_id)] = {}

            for func_A, func_B in tqdm(
                permutations(function_pool, 2),
                ascii=True,
                leave=True,
                desc="eBPF Combinations Checked",
                total=len(function_pool) ** 2,
            ):
                # 双重检查：确保两个函数都是eBPF函数
                if not (
                    _is_ebpf_function(str(func_A)) and _is_ebpf_function(str(func_B))
                ):
                    continue

                if check_same_level(
                    func_A,
                    func_B,
                    process_id,
                    thread_id,
                    function_id_map,
                    function_times,
                ):
                    if func_A not in same_level_func[(process_id, thread_id)]:
                        same_level_func[(process_id, thread_id)][func_A] = []
                    same_level_func[(process_id, thread_id)][func_A].append(func_B)
                    valid_relations[(func_A, func_B)] = True

        # 缓存结果
        self.trace.same_level_func_cover = same_level_func
        self.trace.valid_relations_cover = valid_relations
        self.trace._ebpf_relations_filtered = True

        print(f"End same level checking - Found {len(valid_relations)} eBPF relations")

        return same_level_func, valid_relations


def _prepare_trace_data(trace: TracePandas) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """统一的trace数据准备方法（只处理eBPF函数）"""
    cache = TraceCache(trace)
    function_times, function_id_map, listed_events = (
        cache.get_or_compute_function_data()
    )
    same_level_func, valid_relations = cache.get_or_compute_relations()
    return (
        function_times,
        function_id_map,
        listed_events,
        same_level_func,
        valid_relations,
    )


def _validate_trace_input(trace) -> TracePandas:
    """验证输入trace"""
    if not isinstance(trace, TracePandas):
        raise TypeError("Expected TracePandas instance")
    return trace


def _get_relevant_function_pairs(inv_params: List[APIParam]) -> List[Tuple[str, str]]:
    """获取不变式中相关的函数对（只包含eBPF函数）"""
    pairs = []
    for i in range(len(inv_params) - 1):
        func_A = inv_params[i].api_full_name
        func_B = inv_params[i + 1].api_full_name

        # 确保两个函数都是eBPF函数
        if _is_ebpf_function(func_A) and _is_ebpf_function(func_B):
            pairs.append((func_A, func_B))
    return pairs


def _create_example(event: Dict[str, Any]) -> Example:
    """创建示例对象"""
    example = Example()
    example.add_group(Constants.EXP_GROUP_NAME, [event])
    return example


def _process_function_pair_examples(
    func_A: str,
    func_B: str,
    events_list: List[Dict],
    same_level_func: Dict,
    process_thread_id: Tuple[str, str],
) -> Tuple[List[Example], List[Example]]:
    """处理单个函数对的示例生成"""
    positive_examples = []
    negative_examples = []

    # 确保处理的是eBPF函数
    if not (_is_ebpf_function(func_A) and _is_ebpf_function(func_B)):
        return positive_examples, negative_examples

    if func_B not in same_level_func[process_thread_id]:
        return positive_examples, negative_examples

    if func_A not in same_level_func[process_thread_id][func_B]:
        # 所有B调用都是负例
        for event in events_list:
            if (
                event["type"] == Constants.FUNCTION_CALL_PRE
                and event["function"] == func_B
                and _is_ebpf_function(event["function"])  # 双重检查
            ):
                negative_examples.append(_create_example(event))
        return positive_examples, negative_examples

    # 检查覆盖关系
    flag_A = None
    for event in events_list:
        if event["type"] != Constants.FUNCTION_CALL_PRE:
            continue

        # 只处理eBPF函数的事件
        if not _is_ebpf_function(event["function"]):
            continue

        if func_A == event["function"]:
            flag_A = event["time"]
        elif func_B == event["function"]:
            example = _create_example(event)
            if flag_A is None:
                negative_examples.append(example)
            else:
                positive_examples.append(example)
                flag_A = None

    return positive_examples, negative_examples


def _add_examples_to_hypotheses(
    hypothesis_dict: Dict,
    listed_events: Dict,
    same_level_func: Dict,
    valid_relations: Dict,
) -> None:
    """为假设添加示例（只处理eBPF函数）"""
    print("Start adding examples (eBPF functions only)...")

    ebpf_relations = {
        (func_A, func_B): True
        for (func_A, func_B) in valid_relations.keys()
        if _is_ebpf_function(func_A) and _is_ebpf_function(func_B)
    }

    for (process_id, thread_id), events_list in tqdm(
        listed_events.items(), ascii=True, leave=True, desc="Group"
    ):
        for func_A, func_B in tqdm(ebpf_relations.keys(), desc="eBPF Function Pair"):
            if (func_A, func_B) not in hypothesis_dict:
                continue

            positive_examples, negative_examples = _process_function_pair_examples(
                func_A, func_B, events_list, same_level_func, (process_id, thread_id)
            )

            # 添加示例到假设
            for example in positive_examples:
                hypothesis_dict[(func_A, func_B)].positive_examples.add_example(example)
            for example in negative_examples:
                hypothesis_dict[(func_A, func_B)].negative_examples.add_example(example)

    print("End adding examples")


def _check_function_pair_invariant(
    func_A: str,
    func_B: str,
    events_list: List[Dict],
    same_level_func: Dict,
    process_thread_id: Tuple[str, str],
    inv: Invariant,
    trace: Trace,
) -> Tuple[bool, bool, List[Dict]]:
    """检查单个函数对的不变式（只处理eBPF函数）

    Returns:
        (check_passed, triggered, violation_trace)
    """
    # 确保处理的是eBPF函数
    if not (_is_ebpf_function(func_A) and _is_ebpf_function(func_B)):
        return True, False, []

    if func_B not in same_level_func[process_thread_id]:
        return True, False, []

    if func_A not in same_level_func[process_thread_id][func_B]:
        # 没有A调用，所有B都应该无效
        for event in events_list:
            if (
                event["type"] == Constants.FUNCTION_CALL_PRE
                and event["function"] == func_B
                and _is_ebpf_function(event["function"])  # 确保是eBPF函数
            ):
                if inv.precondition.verify([event], Constants.EXP_GROUP_NAME, trace):
                    return False, True, [event]
        return True, False, []

    # 检查覆盖关系
    unmatched_A_exist = False
    for event in events_list:
        if event["type"] != Constants.FUNCTION_CALL_PRE:
            continue

        # 只处理eBPF函数的事件
        if not _is_ebpf_function(event["function"]):
            continue

        if func_A == event["function"]:
            unmatched_A_exist = True
        elif func_B == event["function"]:
            if inv.precondition.verify([event], Constants.EXP_GROUP_NAME, trace):
                if not unmatched_A_exist:
                    return False, True, [event]
                unmatched_A_exist = False

    return True, False, []


# 其余的辅助函数保持不变
def is_complete_subgraph(
    path: List[APIParam], new_node: APIParam, graph: Dict[APIParam, List[APIParam]]
) -> bool:
    """Check if adding new_node to path forms a complete (directed) graph."""
    for node in path:
        if new_node not in graph[node]:
            return False
    return True


def merge_relations(pairs: List[Tuple[APIParam, APIParam]]) -> List[List[APIParam]]:
    graph: Dict[APIParam, List[APIParam]] = {}
    indegree: Dict[APIParam, int] = {}
    for a, b in pairs:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)

        if b in indegree:
            indegree[b] += 1
        else:
            indegree[b] = 1

        if a not in indegree:
            indegree[a] = 0

    start_nodes: List[APIParam] = [node for node in indegree if indegree[node] == 0]
    paths: List[List[APIParam]] = []

    def add_path(new_path: List[APIParam]) -> None:
        nonlocal paths
        paths.append(new_path)

    def dfs(node: APIParam, path: List[APIParam], visited: Set[APIParam]) -> None:
        path.append(node)
        visited.add(node)
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited and is_complete_subgraph(
                    path, neighbor, graph
                ):
                    dfs(neighbor, path, visited)
        if not graph.get(node):
            add_path(path.copy())
        path.pop()
        visited.remove(node)

    for start_node in start_nodes:
        dfs(start_node, [], set())

    return paths


class CUDARelation(Relation):
    """CUDARelation is a relation that checks if one function covers another function.
    Only processes eBPF functions (functions containing '[eBPF]' in their names).

    say function A and function B are two eBPF functions in the trace, we say function A covers function B when
    every time function B is called, a function A invocation exists before it.
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        """Generate hypothesis for eBPF functions only."""
        try:
            trace = _validate_trace_input(trace)

            # 统一的数据准备（只处理eBPF函数）
            (
                function_times,
                function_id_map,
                listed_events,
                same_level_func,
                valid_relations,
            ) = _prepare_trace_data(trace)

            if not valid_relations:
                logging.getLogger(__name__).warning("No valid eBPF relations found")
                return []

            # 再次过滤，确保只有eBPF函数对
            ebpf_valid_relations = {
                (func_A, func_B): True
                for (func_A, func_B) in valid_relations.keys()
                if _is_ebpf_function(func_A) and _is_ebpf_function(func_B)
            }

            if not ebpf_valid_relations:
                logging.getLogger(__name__).warning(
                    "No valid eBPF function pairs found"
                )
                return []

            # 生成假设
            print(
                f"Start generating hypo for {len(ebpf_valid_relations)} eBPF function pairs..."
            )
            hypothesis_with_examples = {
                (func_A, func_B): Hypothesis(
                    invariant=Invariant(
                        relation=CUDARelation,
                        params=[APIParam(func_A), APIParam(func_B)],
                        precondition=None,
                        text_description=f"eBPF FunctionCoverRelation between {func_A} and {func_B}",
                    ),
                    positive_examples=ExampleList({Constants.EXP_GROUP_NAME}),
                    negative_examples=ExampleList({Constants.EXP_GROUP_NAME}),
                )
                for (func_A, func_B) in ebpf_valid_relations.keys()
            }
            print("End generating hypo")

            # 添加示例
            _add_examples_to_hypotheses(
                hypothesis_with_examples,
                listed_events,
                same_level_func,
                ebpf_valid_relations,
            )

            return list(hypothesis_with_examples.values())

        except Exception as e:
            logging.getLogger(__name__).error(f"Error in generate_hypothesis: {e}")
            return []

    @staticmethod
    def collect_examples(trace, hypothesis):
        """Generate examples for a hypothesis on trace (eBPF functions only)."""
        try:
            trace = _validate_trace_input(trace)

            # 统一的数据准备
            (
                function_times,
                function_id_map,
                listed_events,
                same_level_func,
                valid_relations,
            ) = _prepare_trace_data(trace)

            inv = hypothesis.invariant
            function_pairs = _get_relevant_function_pairs(inv.params)

            # 过滤出eBPF函数
            ebpf_function_pairs = [
                (func_A, func_B)
                for func_A, func_B in function_pairs
                if _is_ebpf_function(func_A) and _is_ebpf_function(func_B)
            ]

            if len(ebpf_function_pairs) == 0:
                print(
                    "No relevant eBPF function calls found in the trace, skipping the collecting"
                )
                return

            print(
                f"Starting collecting iteration for {len(ebpf_function_pairs)} eBPF function pairs..."
            )
            for func_A, func_B in tqdm(ebpf_function_pairs):
                for (process_id, thread_id), events_list in listed_events.items():
                    positive_examples, negative_examples = (
                        _process_function_pair_examples(
                            func_A,
                            func_B,
                            events_list,
                            same_level_func,
                            (process_id, thread_id),
                        )
                    )

                    # 添加示例到假设
                    for example in positive_examples:
                        hypothesis.positive_examples.add_example(example)
                    for example in negative_examples:
                        hypothesis.negative_examples.add_example(example)

            print("End collecting iteration...")

        except Exception as e:
            logging.getLogger(__name__).error(f"Error in collect_examples: {e}")

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for eBPF functions only."""
        try:
            all_hypotheses = CUDARelation.generate_hypothesis(trace)

            if_merge = True

            print("Start precondition inference...")
            failed_hypothesis = []
            for hypothesis in all_hypotheses.copy():
                preconditions = find_precondition(hypothesis, [trace])
                if preconditions is not None:
                    hypothesis.invariant.precondition = preconditions
                else:
                    failed_hypothesis.append(
                        FailedHypothesis(hypothesis, "Precondition not found")
                    )
                    all_hypotheses.remove(hypothesis)
            print("End precondition inference")

            if not if_merge:
                return ([hypo.invariant for hypo in all_hypotheses], failed_hypothesis)

            # 合并不变式
            print("Start merging invariants...")
            relation_pool: Dict[
                GroupedPreconditions | None, List[Tuple[APIParam, APIParam]]
            ] = {}

            for hypothesis in all_hypotheses:
                param_A = hypothesis.invariant.params[0]
                param_B = hypothesis.invariant.params[1]

                # 确保参数是eBPF函数
                if _is_ebpf_function(param_A.api_full_name) and _is_ebpf_function(
                    param_B.api_full_name
                ):

                    if hypothesis.invariant.precondition not in relation_pool:
                        relation_pool[hypothesis.invariant.precondition] = []
                    relation_pool[hypothesis.invariant.precondition].append(
                        (param_A, param_B)
                    )

            merged_relations: Dict[
                GroupedPreconditions | None, List[List[APIParam]]
            ] = {}

            for key, values in tqdm(
                relation_pool.items(), desc="Merging eBPF Invariants"
            ):
                merged_relations[key] = merge_relations(values)

            merged_invariants = []
            for key, merged_values in merged_relations.items():
                for merged_value in merged_values:
                    new_invariant = Invariant(
                        relation=CUDARelation,
                        params=list(merged_value),
                        precondition=key,
                        text_description="Merged eBPF FunctionCoverRelation in Ordered List",
                    )
                    merged_invariants.append(new_invariant)
            print(
                f"End merging invariants - Generated {len(merged_invariants)} eBPF invariants"
            )

            return merged_invariants, failed_hypothesis

        except Exception as e:
            logging.getLogger(__name__).error(f"Error in infer: {e}")
            return [], []

    @staticmethod
    def evaluate(value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        return True

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        """Check invariant on trace (eBPF functions only)."""
        try:
            if inv.precondition is None:
                raise ValueError("Invariant should have a precondition.")

            trace = _validate_trace_input(trace)

            # 统一的数据准备
            (
                function_times,
                function_id_map,
                listed_events,
                same_level_func,
                valid_relations,
            ) = _prepare_trace_data(trace)

            # 检查是否有相关的eBPF函数
            function_pairs = _get_relevant_function_pairs(inv.params)
            ebpf_function_pairs = [
                (func_A, func_B)
                for func_A, func_B in function_pairs
                if _is_ebpf_function(func_A) and _is_ebpf_function(func_B)
            ]

            if len(ebpf_function_pairs) == 0:
                print(
                    "No relevant eBPF function calls found in the trace, skipping the checking"
                )
                return CheckerResult(
                    trace=None, invariant=inv, check_passed=True, triggered=False
                )

            inv_triggered = False

            print(
                f"Starting checking iteration for {len(ebpf_function_pairs)} eBPF function pairs..."
            )
            for func_A, func_B in tqdm(ebpf_function_pairs):
                for (process_id, thread_id), events_list in listed_events.items():
                    check_passed, triggered, violation_trace = (
                        _check_function_pair_invariant(
                            func_A,
                            func_B,
                            events_list,
                            same_level_func,
                            (process_id, thread_id),
                            inv,
                            trace,
                        )
                    )

                    if triggered:
                        inv_triggered = True

                    if not check_passed:
                        return CheckerResult(
                            trace=violation_trace,
                            invariant=inv,
                            check_passed=False,
                            triggered=True,
                        )

            return CheckerResult(
                trace=None, invariant=inv, check_passed=True, triggered=inv_triggered
            )

        except Exception as e:
            logging.getLogger(__name__).error(f"Error in static_check_all: {e}")
            return CheckerResult(
                trace=None, invariant=inv, check_passed=False, triggered=False
            )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return ["function"]
