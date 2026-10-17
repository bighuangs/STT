"""用静态导入图守卫仓库架构边界与迁移债务。"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWN_DEBT_PATH = Path(__file__).parent / "fixtures" / "known_debt.json"


@dataclass(frozen=True, slots=True)
class ModuleSource:
    """保存一个 Python 模块的物理位置与语法树。"""

    name: str
    path: Path
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class ImportIndex:
    """保存仓库内模块及其静态导入关系。"""

    modules: dict[str, ModuleSource]
    imports: dict[str, frozenset[str]]


def _is_same_or_child(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _python_paths() -> list[Path]:
    paths: list[Path] = []
    for root_name in ("app", "evals"):
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(paths)


def _resolve_from_import(
    source: ModuleSource,
    node: ast.ImportFrom,
    known_modules: frozenset[str],
) -> set[str]:
    if node.level:
        package_parts = source.name.split(".")
        if source.path.name != "__init__.py":
            package_parts.pop()
        keep = max(0, len(package_parts) - (node.level - 1))
        base_parts = package_parts[:keep]
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)
    else:
        base = node.module or ""

    targets: set[str] = set()
    for alias in node.names:
        candidate = f"{base}.{alias.name}" if base else alias.name
        if alias.name != "*" and candidate in known_modules:
            targets.add(candidate)
        elif base:
            targets.add(base)
        else:
            targets.add(candidate)
    return targets


@lru_cache(maxsize=1)
def _import_index() -> ImportIndex:
    modules: dict[str, ModuleSource] = {}
    for path in _python_paths():
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8-sig"),
                filename=str(path),
            )
        except (OSError, SyntaxError) as exc:
            raise AssertionError(f"无法解析 Python 模块 {path}: {exc}") from exc
        name = _module_name(path)
        modules[name] = ModuleSource(name=name, path=path, tree=tree)

    known_modules = frozenset(modules)
    imports: dict[str, frozenset[str]] = {}
    for name, source in modules.items():
        targets: set[str] = set()
        for node in ast.walk(source.tree):
            if isinstance(node, ast.Import):
                targets.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets.update(_resolve_from_import(source, node, known_modules))
        imports[name] = frozenset(targets)
    return ImportIndex(modules=modules, imports=imports)


@lru_cache(maxsize=1)
def _known_debt() -> dict[str, object]:
    payload = json.loads(KNOWN_DEBT_PATH.read_text(encoding="utf-8"))
    required = {
        "application_outward_edges",
        "cyclic_edges",
        "import_time_constructions",
        "legacy_modules",
        "runtime_legacy_edges",
    }
    missing = required.difference(payload)
    assert not missing, f"架构债务基线缺少字段: {sorted(missing)}"
    return payload


def _format_edges(edges: Iterable[str]) -> str:
    return "\n".join(f"  - {edge}" for edge in sorted(edges))


def _scoped_edges(scope: str) -> set[str]:
    index = _import_index()
    return {
        f"{source} -> {target}"
        for source, targets in index.imports.items()
        if _is_same_or_child(source, scope)
        for target in targets
    }


def _forbidden_edges(scope: str, forbidden: Iterable[str]) -> set[str]:
    forbidden_packages = tuple(forbidden)
    return {
        edge
        for edge in _scoped_edges(scope)
        if any(
            _is_same_or_child(edge.partition(" -> ")[2], package) for package in forbidden_packages
        )
    }


def _assert_scope_uses_only(
    scope: str,
    allowed_internal_packages: Iterable[str],
) -> None:
    violations = _scope_dependency_violations(
        scope,
        allowed_internal_packages,
    )
    assert not violations, f"{scope} 出现未授权的内部依赖:\n{_format_edges(violations)}"


def _scope_dependency_violations(
    scope: str,
    allowed_internal_packages: Iterable[str],
) -> set[str]:
    allowed = tuple(allowed_internal_packages)
    violations: set[str] = set()
    for edge in _scoped_edges(scope):
        target = edge.partition(" -> ")[2]
        if _is_same_or_child(target, "evals"):
            violations.add(edge)
            continue
        if not _is_same_or_child(target, "app"):
            continue
        if not any(_is_same_or_child(target, package) for package in allowed):
            violations.add(edge)
    return violations


def _is_public_application_contract(target: str) -> bool:
    if target == "app.application":
        return True
    if target == "app.application.evaluation_target":
        return True
    if not target.startswith("app.application."):
        return False
    relative_parts = target.removeprefix("app.application.").split(".")
    public_segments = {"api", "contracts", "dto", "dtos", "ports", "public"}
    if public_segments.intersection(relative_parts):
        return True
    return relative_parts[-1].endswith(("_api", "_contract", "_contracts", "_dto", "_ports"))


def _strongly_connected_components(
    graph: dict[str, frozenset[str]],
) -> list[frozenset[str]]:
    """使用 Tarjan 算法返回仓库内部导入图的强连通分量。"""

    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[frozenset[str]] = []

    def visit(module: str) -> None:
        nonlocal index
        indexes[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for dependency in sorted(graph[module]):
            if dependency not in indexes:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[module] = min(lowlinks[module], indexes[dependency])

        if lowlinks[module] != indexes[module]:
            return
        component: set[str] = set()
        while stack:
            dependency = stack.pop()
            on_stack.remove(dependency)
            component.add(dependency)
            if dependency == module:
                break
        components.append(frozenset(component))

    for module in sorted(graph):
        if module not in indexes:
            visit(module)
    return components


def _cyclic_edges() -> set[str]:
    index = _import_index()
    internal_graph = {
        source: frozenset(target for target in targets if target in index.modules)
        for source, targets in index.imports.items()
    }
    cyclic: set[str] = set()
    for component in _strongly_connected_components(internal_graph):
        if len(component) == 1:
            only = next(iter(component))
            if only not in internal_graph[only]:
                continue
        for source in component:
            for target in internal_graph[source].intersection(component):
                cyclic.add(f"{source} -> {target}")
    return cyclic


def _terminal_attribute(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _import_aliases(source: ModuleSource) -> tuple[set[str], set[str]]:
    fastapi_names: set[str] = set()
    state_graph_names: set[str] = set()
    for node in source.tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "fastapi":
                fastapi_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "FastAPI"
                )
            if node.module in {"langgraph", "langgraph.graph"}:
                state_graph_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "StateGraph"
                )
    return fastapi_names, state_graph_names


def _is_named_call(node: ast.AST, names: set[str], attribute: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in names
    return isinstance(node.func, ast.Attribute) and node.func.attr == attribute


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    values = [test.left, *test.comparators]
    has_name = any(isinstance(value, ast.Name) and value.id == "__name__" for value in values)
    has_main = any(
        isinstance(value, ast.Constant) and value.value == "__main__" for value in values
    )
    return has_name and has_main and isinstance(test.ops[0], ast.Eq)


class _ImportTimeConstructionVisitor(ast.NodeVisitor):
    """识别模块导入时执行的框架对象构造。"""

    def __init__(self, source: ModuleSource) -> None:
        self.source = source
        self.fastapi_names, self.state_graph_names = _import_aliases(source)
        self.graph_variables: set[str] = set()
        self.occurrences: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def _remember_graph_assignment(
        self,
        targets: Iterable[ast.expr],
        value: ast.expr | None,
    ) -> None:
        if value is None or not _is_named_call(value, self.state_graph_names, "StateGraph"):
            return
        for target in targets:
            if isinstance(target, ast.Name):
                self.graph_variables.add(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._remember_graph_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._remember_graph_assignment((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_named_call(node, self.fastapi_names, "FastAPI"):
            self.occurrences.append(("FastAPI", node.lineno))
        if _is_named_call(node, self.state_graph_names, "StateGraph"):
            self.occurrences.append(("StateGraph", node.lineno))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "compile":
            receiver = node.func.value
            is_graph_receiver = (
                isinstance(receiver, ast.Name) and receiver.id in self.graph_variables
            ) or _is_named_call(receiver, self.state_graph_names, "StateGraph")
            if is_graph_receiver:
                self.occurrences.append(("StateGraph.compile", node.lineno))
        self.generic_visit(node)


def _import_time_constructions() -> tuple[Counter[str], dict[str, list[int]]]:
    counts: Counter[str] = Counter()
    locations: dict[str, list[int]] = {}
    for name, source in _import_index().modules.items():
        if _is_same_or_child(name, "app.bootstrap"):
            continue
        visitor = _ImportTimeConstructionVisitor(source)
        visitor.visit(source.tree)
        for kind, line in visitor.occurrences:
            key = f"{name}::{kind}"
            counts[key] += 1
            locations.setdefault(key, []).append(line)
    return counts, locations


def test_known_debt_baseline_is_explicit_and_machine_readable() -> None:
    payload = _known_debt()
    assert payload.get("schema_version") == 1
    assert isinstance(payload.get("policy"), str) and payload["policy"]

    for field in (
        "application_outward_edges",
        "cyclic_edges",
        "legacy_modules",
        "runtime_legacy_edges",
    ):
        entries = payload[field]
        assert isinstance(entries, list), f"{field} 必须是 JSON 数组"
        assert entries == sorted(entries), f"{field} 必须保持排序以便审查 diff"
        assert len(entries) == len(set(entries)), f"{field} 不得含重复债务"

    constructions = payload["import_time_constructions"]
    assert isinstance(constructions, dict)
    assert list(constructions) == sorted(constructions), (
        "import_time_constructions 必须保持排序以便审查 diff"
    )
    assert all(type(count) is int and count > 0 for count in constructions.values()), (
        "导入期构造债务计数必须是正整数"
    )


def test_tarjan_analyzer_detects_only_cyclic_components() -> None:
    graph = {
        "cycle.a": frozenset({"cycle.b"}),
        "cycle.b": frozenset({"cycle.a", "leaf"}),
        "leaf": frozenset(),
    }
    components = _strongly_connected_components(graph)
    assert frozenset({"cycle.a", "cycle.b"}) in components
    assert frozenset({"leaf"}) in components


def test_import_time_analyzer_ignores_lazy_function_bodies() -> None:
    tree = ast.parse(
        """
from fastapi import FastAPI
from langgraph.graph import StateGraph

graph = StateGraph(dict)
compiled = graph.compile()
app = FastAPI()

def build_lazily():
    lazy_graph = StateGraph(dict)
    return lazy_graph.compile()
"""
    )
    source = ModuleSource(
        name="synthetic.module",
        path=PROJECT_ROOT / "synthetic" / "module.py",
        tree=tree,
    )
    visitor = _ImportTimeConstructionVisitor(source)
    visitor.visit(tree)
    assert Counter(kind for kind, _ in visitor.occurrences) == Counter(
        {"FastAPI": 1, "StateGraph": 1, "StateGraph.compile": 1}
    )


def test_domain_does_not_depend_on_outer_application_layers() -> None:
    violations = {
        edge
        for edge in _scoped_edges("app.domain")
        if (
            _is_same_or_child(edge.partition(" -> ")[2], "app")
            and not _is_same_or_child(edge.partition(" -> ")[2], "app.domain")
        )
        or _is_same_or_child(edge.partition(" -> ")[2], "evals")
    }
    assert not violations, f"Domain 层不得依赖任何外层 app 包:\n{_format_edges(violations)}"


def test_application_outward_dependencies_do_not_exceed_baseline() -> None:
    forbidden = {
        "app.adapters",
        "app.agent_system",
        "app.api",
        "app.bootstrap",
        "app.config",
        "app.evaluation",
        "app.infra",
        "app.infrastructure",
        "app.process",
        "app.rag",
        "app.runtime",
        "app.security",
        "app.shared",
        "app.worker",
        "evals",
    }
    current = _forbidden_edges("app.application", forbidden)
    baseline = set(_known_debt()["application_outward_edges"])
    unexpected = current.difference(baseline)
    assert not unexpected, (
        f"Application 层新增了对具体实现或外层入口的逆向依赖:\n{_format_edges(unexpected)}"
    )


def test_new_target_packages_follow_strict_dependency_direction() -> None:
    _assert_scope_uses_only("app.config", {"app.config"})
    _assert_scope_uses_only("app.foundation", {"app.foundation"})
    _assert_scope_uses_only(
        "app.adapters.inbound",
        {
            "app.adapters.inbound",
            "app.application",
            "app.contracts",
            "app.domain",
        },
    )
    _assert_scope_uses_only(
        "app.adapters.outbound",
        {
            "app.adapters.outbound",
            "app.application",
            "app.config",
            "app.contracts",
            "app.domain",
        },
    )
    _assert_scope_uses_only(
        "app.agent_system",
        {
            "app.agent_system",
            "app.application",
            "app.contracts",
            "app.domain",
        },
    )
    runtime_violations = _scope_dependency_violations(
        "app.runtime",
        {
            "app.agent_system",
            "app.application",
            "app.contracts",
            "app.domain",
            "app.runtime",
        },
    )
    runtime_baseline = set(_known_debt()["runtime_legacy_edges"])
    unexpected_runtime_dependencies = runtime_violations.difference(runtime_baseline)
    assert not unexpected_runtime_dependencies, (
        f"Runtime 新增了迁移缝之外的内部依赖:\n{_format_edges(unexpected_runtime_dependencies)}"
    )

    infrastructure_entrypoint_dependencies = _forbidden_edges(
        "app.infrastructure",
        {
            "app.adapters.inbound",
            "app.api",
            "app.bootstrap",
            "app.worker",
            "evals",
        },
    )
    assert not infrastructure_entrypoint_dependencies, (
        "Infrastructure 迁移适配器不得反向依赖入口或组合根:\n"
        f"{_format_edges(infrastructure_entrypoint_dependencies)}"
    )


def test_agents_do_not_import_another_agents_private_package() -> None:
    violations: set[str] = set()
    prefix = "app.agent_system.agents."
    for edge in _scoped_edges("app.agent_system.agents"):
        source, _, target = edge.partition(" -> ")
        if not source.startswith(prefix) or not target.startswith(prefix):
            continue
        source_parts = source.removeprefix(prefix).split(".")
        target_parts = target.removeprefix(prefix).split(".")
        if source_parts[0] != target_parts[0]:
            violations.add(edge)
    assert not violations, (
        "Agent 只能通过公共 contract 协作，不得导入另一个 Agent 的私有包:\n"
        f"{_format_edges(violations)}"
    )


def test_evals_only_depend_on_stable_product_contracts() -> None:
    violations: set[str] = set()
    for edge in _scoped_edges("evals"):
        target = edge.partition(" -> ")[2]
        if not _is_same_or_child(target, "app"):
            continue
        if _is_same_or_child(target, "app.domain"):
            continue
        if _is_same_or_child(target, "app.contracts"):
            continue
        if _is_public_application_contract(target):
            continue
        violations.add(edge)
    assert not violations, (
        "评测代码只能依赖 Domain、稳定 contract 或 Application 公开接口:\n"
        f"{_format_edges(violations)}"
    )


def test_internal_import_graph_has_no_new_cycles() -> None:
    current = _cyclic_edges()
    baseline = set(_known_debt()["cyclic_edges"])
    unexpected = current.difference(baseline)
    assert not unexpected, (
        f"内部导入图出现了新的循环边；请消除 SCC，不要扩充债务基线:\n{_format_edges(unexpected)}"
    )


def test_framework_objects_are_not_constructed_at_import_time_outside_bootstrap() -> None:
    current, locations = _import_time_constructions()
    raw_baseline = _known_debt()["import_time_constructions"]
    baseline = Counter({str(key): int(value) for key, value in raw_baseline.items()})
    unexpected: list[str] = []
    for key, count in sorted(current.items()):
        excess = count - baseline[key]
        if excess > 0:
            lines = ", ".join(str(line) for line in locations[key])
            unexpected.append(f"{key} 共 {count} 处（行 {lines}），基线允许 {baseline[key]} 处")
    assert not unexpected, (
        f"FastAPI/StateGraph 只能在 bootstrap 组合根导入期构造:\n{_format_edges(unexpected)}"
    )


def test_legacy_root_modules_do_not_increase() -> None:
    legacy_roots = {
        "app.api",
        "app.evaluation",
        "app.infra",
        "app.process",
        "app.rag",
        "app.security",
        "app.shared",
        "app.worker",
    }
    current = {
        module
        for module in _import_index().modules
        if any(_is_same_or_child(module, root) for root in legacy_roots)
    }
    baseline = set(_known_debt()["legacy_modules"])
    unexpected = current.difference(baseline)
    assert not unexpected, f"旧根目录只能迁出或删除，不得新增模块:\n{_format_edges(unexpected)}"
