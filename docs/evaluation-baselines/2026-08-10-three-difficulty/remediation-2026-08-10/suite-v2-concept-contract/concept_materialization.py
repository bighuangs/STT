"""Shared deterministic concept-group materialization rules.

The reviewed-suite merger and the independent validator intentionally import
this module instead of maintaining two copies of the discriminative-variant
gate.  Cross-concept ambiguity is evaluated with the same matcher used by the
production evaluator: a variant is removed when a standalone variant from a
different all-of concept would satisfy it.  This prevents one answer fragment
from receiving credit for multiple authored concepts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

from app.evaluation.metrics import (  # noqa: E402
    CONCEPT_VARIANT_MATCHER_POLICY_VERSION,
    concept_variant_covered,
    concept_variant_covered_legacy,
)


MATERIALIZATION_POLICY_VERSION = (
    "stt.eval.concept-groups.discriminative-variant.v6-global-subset"
)
V5_MATERIALIZATION_POLICY_VERSION = (
    "stt.eval.concept-groups.discriminative-variant.v5-cjk-sibling-evidence"
)
LEGACY_MATERIALIZATION_POLICY_VERSION = (
    "stt.eval.concept-groups.discriminative-variant.v4-meta-evidence-matcher"
)
PRODUCTION_MATCHER = "app.evaluation.metrics.concept_variant_covered"
PRODUCTION_MATCHER_POLICY_VERSION = CONCEPT_VARIANT_MATCHER_POLICY_VERSION
SUBSET_SOLVER_ALGORITHM = (
    "collision-components-bounded-exact-independent-subset-v1"
)
SUBSET_SOLVER_OBJECTIVE = "all_groups_nonempty_then_maximize_total_variants"
SUBSET_SOLVER_TIE_BREAK = "prefer_earlier_group_and_variant_input_order"
SUBSET_SOLVER_MAX_TOTAL_VERTICES = 160
SUBSET_SOLVER_MAX_COMPONENT_VERTICES = 22
SUBSET_SOLVER_MAX_COMPONENT_STATES = 250_000
SUBSET_SOLVER_MAX_TOTAL_STATES = 500_000
SUBSET_SOLVER_POLICY = {
    "algorithm": SUBSET_SOLVER_ALGORITHM,
    "objective": SUBSET_SOLVER_OBJECTIVE,
    "tie_break": SUBSET_SOLVER_TIE_BREAK,
    "max_total_vertices": SUBSET_SOLVER_MAX_TOTAL_VERTICES,
    "max_component_vertices": SUBSET_SOLVER_MAX_COMPONENT_VERTICES,
    "max_component_states": SUBSET_SOLVER_MAX_COMPONENT_STATES,
    "max_total_states": SUBSET_SOLVER_MAX_TOTAL_STATES,
}

_GENERIC_WEAK_VARIANTS = {
    "点击",
    "单击",
    "选择",
    "打开",
    "关闭",
    "查看",
    "完成",
    "按",
    "按下",
    "执行",
    "操作",
    "设置",
    "按钮",
    "选项",
    "菜单",
    "图标",
    "步骤",
    "信息",
    "内容",
    "安装",
    "添加",
    "键",
    "或",
    "打印",
    "开启",
    "进入",
    "访问",
}
_GENERIC_ACTION_ONLY_RE = re.compile(
    r"^(?:点击|单击|选择|打开|关闭|查看|完成|按|按下|执行|操作|设置|勾选|"
    r"安装|添加|打印|开启|进入|访问)"
    r"(?:按钮|选项|菜单|图标|步骤|信息|内容|设置)?$"
)
_GENERIC_COMPOSITION_COMPONENT_RE = re.compile(
    r"(?:点击|单击|选择|打开|关闭|查看|完成|按下|按|执行|操作|设置|勾选|"
    r"安装|添加|打印|开启|进入|访问|按钮|选项|菜单|图标|步骤|信息|内容|"
    r"键|或|以及|并且|然后|并|该|此|相应|对应)"
)
_QUERY_META_DIRECTIVE_ONLY_RE = re.compile(
    r"^(?:请)?(?:"
    r"(?:按(?:照)?(?:正确)?顺序|依次|按步骤)"
    r"(?:说明|列出|回答|操作)?(?:全部|所有)?(?:操作)?(?:步骤?)?"
    r"|(?:说明|列出|回答)(?:全部|所有|具体|相关)?(?:操作)?(?:步骤?)?"
    r"|列出(?:全部|所有)(?:操作)?(?:步骤?)?"
    r"|结合(?:证据|资料|文档)(?:说明|回答)?"
    r"|具体(?:操作)?路径"
    r"|分别(?:说明|列出|回答)(?:各项|问题|各个问题|两个问题)?"
    r")$",
    re.IGNORECASE,
)
ABSTENTION_SENTINEL_CANONICAL_TEXT = "资料不足，无法依据已授权证据确定答案"
ABSTENTION_SENTINEL_VARIANTS = (
    "资料不足",
    "未提供相关信息",
    "无法确定",
)
ABSTENTION_SENTINEL_POLARITY = {
    "required": "abstain_without_factual_claims",
    "cues": ["资料不足", "无法确定"],
}
_ABSTENTION_SENTINEL_TEXTS = frozenset(
    (ABSTENTION_SENTINEL_CANONICAL_TEXT, *ABSTENTION_SENTINEL_VARIANTS)
)
_LEGACY_ABSTENTION_CUE_RE = re.compile(
    r"(?:未提供|没有提供|未给出|未说明|未记载|未包含|无相关|无.*信息|没有.*信息|"
    r"无法|不能确定|不可确定|不足以|缺少|缺乏|不包含|"
    r"not provided|cannot determine|insufficient)",
    re.IGNORECASE,
)


def abstention_variant_is_pure_refusal(value: Any) -> bool:
    """Accept only one frozen, reviewer-independent abstention sentinel.

    V2 deliberately does not parse model-authored natural language.  A saved
    recovery must first replace the sole unsupported abstention concept with
    these exact strings; this predicate then validates only that projection.
    """

    return str(value or "") in _ABSTENTION_SENTINEL_TEXTS


def abstention_variant_is_refusal(
    value: Any, *, policy: str = "v2_pure_clause"
) -> bool:
    """Evaluate a refusal variant under one explicit, frozen policy."""

    if policy == "v1_substring":
        return bool(_LEGACY_ABSTENTION_CUE_RE.search(str(value or "")))
    if policy == "v2_pure_clause":
        return abstention_variant_is_pure_refusal(value)
    raise ValueError(f"unknown abstention cue policy: {policy}")


def policy_normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def meta_directive_reason(value: str) -> str | None:
    compact = re.sub(r"[\s\W_]+", "", str(value or ""), flags=re.UNICODE).casefold()
    if compact and _QUERY_META_DIRECTIVE_ONLY_RE.fullmatch(compact):
        return "query_meta_directive_without_answer_fact"
    return None


def variant_weak_reason(
    variant: str,
    *,
    critical_values: set[str],
    evidence_tokens: set[str],
    evidence_source_text: str = "",
    matcher: Callable[[str, str], bool] = concept_variant_covered,
) -> str | None:
    compact = re.sub(r"[\s\W_]+", "", variant, flags=re.UNICODE).casefold()
    if not compact:
        return "empty"
    directive_reason = meta_directive_reason(variant)
    if directive_reason is not None:
        return directive_reason
    generic = {
        policy_normalized(value).replace(" ", "")
        for value in _GENERIC_WEAK_VARIANTS
    }
    if compact in generic:
        return "generic_action_or_meta"
    if _GENERIC_ACTION_ONLY_RE.fullmatch(compact):
        return "generic_action_without_object"
    if not _GENERIC_COMPOSITION_COMPONENT_RE.sub("", compact):
        return "generic_action_or_meta_composition"
    if re.fullmatch(r"[a-z0-9./:+-]{1,10}", compact, flags=re.IGNORECASE):
        numeric_atoms = {
            re.sub(r"[\s\W_]+", "", atom, flags=re.UNICODE).casefold()
            for atom in re.findall(r"\d+(?:[.,]\d+)?", variant)
        }
        critical_support = compact in critical_values or bool(numeric_atoms) and all(
            atom in critical_values for atom in numeric_atoms
        )
        evidence_support = compact in evidence_tokens or matcher(
            evidence_source_text, variant
        )
        if not critical_support or not evidence_support:
            return "short_identifier_or_numeric_without_explicit_support"
    return None


def _collision_witnesses(
    groups: Sequence[dict[str, Any]],
    *,
    matcher=concept_variant_covered,
) -> dict[tuple[int, str], tuple[str, str]]:
    """Return target variants covered by a different concept's standalone text.

    The key identifies the *target* variant that makes production scoring too
    permissive.  Removing the target, instead of the witness, preserves a
    specific valid concept when it merely contains a shorter or semantically
    broader target (for example ``192.168.0.102`` versus ``192.168.0.1``).
    """

    collisions: dict[tuple[int, str], tuple[str, str]] = {}
    for target_index, target_group in enumerate(groups):
        for target_variant in target_group.get("any_of") or []:
            normalized_target = policy_normalized(target_variant)
            for witness_index, witness_group in enumerate(groups):
                if witness_index == target_index:
                    continue
                for witness_variant in witness_group.get("any_of") or []:
                    if matcher(
                        str(witness_variant), str(target_variant)
                    ):
                        collisions[(target_index, normalized_target)] = (
                            str(witness_group.get("concept_key") or ""),
                            str(witness_variant),
                        )
                        break
                if (target_index, normalized_target) in collisions:
                    break
    return collisions


def _maximum_collision_free_subsets(
    groups: Sequence[dict[str, Any]],
    *,
    matcher: Callable[[str, str], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """选择跨组互不覆盖且总variant数最大的确定性非空子集。

    搜索顺序固定为group输入序、variant输入序，并在相同最大数量时优先保留
    更早的variant。任何group无法保留至少一个variant时都严格失败。
    """

    items = [
        (group_index, variant_index, str(variant))
        for group_index, group in enumerate(groups)
        for variant_index, variant in enumerate(group.get("any_of") or [])
    ]
    if len(items) > SUBSET_SOLVER_MAX_TOTAL_VERTICES:
        raise ValueError("collision solver total-vertex cap exceeded")

    total_work_states = 0
    pair_evaluations = 0
    component_search_states = 0
    dynamic_program_states = 0

    def consume_work(*, category: str) -> None:
        nonlocal total_work_states
        total_work_states += 1
        if total_work_states > SUBSET_SOLVER_MAX_TOTAL_STATES:
            raise ValueError(f"collision solver global state cap exceeded during {category}")

    adjacency: list[set[int]] = [set() for _ in items]
    for left_index, (left_group, _left_variant, left) in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            right_group, _right_variant, right = items[right_index]
            if left_group == right_group:
                continue
            consume_work(category="conflict graph construction")
            pair_evaluations += 1
            if matcher(left, right) or matcher(right, left):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    selected_bits = sum(
        1 << (len(items) - 1 - index)
        for index, neighbors in enumerate(adjacency)
        if not neighbors
    )
    initial_group_mask = 0
    for index, neighbors in enumerate(adjacency):
        if not neighbors:
            initial_group_mask |= 1 << items[index][0]

    components: list[list[int]] = []
    unseen = {index for index, neighbors in enumerate(adjacency) if neighbors}
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: list[int] = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    # coverage mask -> (selected count, stable authored-order preference bits)
    dynamic: dict[int, tuple[int, int]] = {
        initial_group_mask: (selected_bits.bit_count(), selected_bits)
    }
    for component in components:
        if len(component) > SUBSET_SOLVER_MAX_COMPONENT_VERTICES:
            raise ValueError("collision component exceeds bounded exact-search cap")
        options: dict[int, tuple[int, int]] = {}
        visited_states = 0

        def enumerate_component(
            position: int, chosen: list[int], group_mask: int, bits: int
        ) -> None:
            nonlocal visited_states, component_search_states
            consume_work(category="component exact search")
            visited_states += 1
            component_search_states += 1
            if visited_states > SUBSET_SOLVER_MAX_COMPONENT_STATES:
                raise ValueError("collision component exact-search state cap exceeded")
            if position == len(component):
                candidate = (len(chosen), bits)
                prior = options.get(group_mask)
                if prior is None or candidate > prior:
                    options[group_mask] = candidate
                return
            item_index = component[position]
            if all(item_index not in adjacency[other] for other in chosen):
                enumerate_component(
                    position + 1,
                    [*chosen, item_index],
                    group_mask | (1 << items[item_index][0]),
                    bits | (1 << (len(items) - 1 - item_index)),
                )
            enumerate_component(position + 1, chosen, group_mask, bits)

        enumerate_component(0, [], 0, 0)
        combined: dict[int, tuple[int, int]] = {}
        for left_mask, (left_count, left_bits) in dynamic.items():
            for right_mask, (right_count, right_bits) in options.items():
                consume_work(category="component dynamic programming")
                dynamic_program_states += 1
                mask = left_mask | right_mask
                candidate = (left_count + right_count, left_bits | right_bits)
                prior = combined.get(mask)
                if prior is None or candidate > prior:
                    combined[mask] = candidate
        dynamic = combined

    required_mask = (1 << len(groups)) - 1
    best = dynamic.get(required_mask)
    if best is None:
        raise ValueError("concept groups have no collision-free nonempty subset")
    best_bits = best[1]
    selected_groups: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        variants: list[str] = []
        for variant in group.get("any_of") or []:
            if best_bits & (1 << (len(items) - 1 - cursor)):
                variants.append(str(variant))
            cursor += 1
        if not variants:  # pragma: no cover - guaranteed by the search invariant
            raise ValueError("concept group lost every variant")
        selected_groups.append(
            {"concept_key": str(group["concept_key"]), "any_of": variants}
        )
    for left_index, left_group in enumerate(selected_groups):
        for right_index, right_group in enumerate(selected_groups):
            if left_index == right_index:
                continue
            for left in left_group["any_of"]:
                for right in right_group["any_of"]:
                    if matcher(left, right) or matcher(right, left):
                        raise ValueError("collision-free subset postcondition failed")
    return selected_groups, {
        **SUBSET_SOLVER_POLICY,
        "component_count": len(components),
        "total_vertices": len(items),
        "pair_evaluations": pair_evaluations,
        "visited_component_states": component_search_states,
        "visited_dynamic_program_states": dynamic_program_states,
        "visited_total_states": total_work_states,
        "completed": True,
    }


def materialize_concept_groups(
    contract: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    abstention_cue_policy: str = "v2_pure_clause",
    policy_version: str = MATERIALIZATION_POLICY_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy_version in {
        MATERIALIZATION_POLICY_VERSION,
        V5_MATERIALIZATION_POLICY_VERSION,
    }:
        production_matcher = concept_variant_covered
        production_matcher_policy = PRODUCTION_MATCHER_POLICY_VERSION
    elif policy_version == LEGACY_MATERIALIZATION_POLICY_VERSION:
        production_matcher = concept_variant_covered_legacy
        production_matcher_policy = "stt.eval.concept-variant-matcher.v3"
    else:
        raise ValueError(f"unknown materialization policy version: {policy_version}")
    constraints = contract.get("critical_constraints") or {}
    critical_values = {
        re.sub(
            r"[\s\W_]+", "", str(item.get("value") or ""), flags=re.UNICODE
        ).casefold()
        for key in ("numerics", "identifiers")
        for item in constraints.get(key) or []
        if str(item.get("value") or "").strip()
    }
    evidence_source_text = "\n".join(
        str(item.get("text") or "") for item in evidence
    )
    evidence_tokens = {
        re.sub(r"[\s\W_]+", "", token, flags=re.UNICODE).casefold()
        for token in re.findall(
            r"[A-Za-z0-9]+(?:[./:+-][A-Za-z0-9]+)*", evidence_source_text
        )
    }
    provisional_groups: list[dict[str, Any]] = []
    concept_keys: set[str] = set()
    removed_variants: list[dict[str, str]] = []
    weak_canonical_count = 0
    if (contract.get("answerability") or {}).get("required") == "abstain":
        for index, concept in enumerate(
            (contract.get("required_concepts") or {}).get("all_of") or []
        ):
            variants = [
                str(concept.get("canonical_text") or ""),
                *(str(value or "") for value in concept.get("any_of") or []),
            ]
            if (
                abstention_cue_policy == "v2_pure_clause"
                and (
                    concept.get("canonical_text")
                    != ABSTENTION_SENTINEL_CANONICAL_TEXT
                    or concept.get("any_of") != list(ABSTENTION_SENTINEL_VARIANTS)
                    or constraints.get("polarity") != ABSTENTION_SENTINEL_POLARITY
                )
            ) or any(
                not abstention_variant_is_refusal(
                    value, policy=abstention_cue_policy
                )
                for value in variants
            ):
                raise ValueError(
                    f"abstain concept[{index}] contains a non-refusal factual variant"
                )
    for concept in (contract.get("required_concepts") or {}).get("all_of") or []:
        concept_key = str(concept.get("concept_id") or "").strip()
        canonical = str(concept.get("canonical_text") or "").strip()
        canonical_meta_reason = meta_directive_reason(canonical)
        if canonical_meta_reason is not None:
            raise ValueError(
                f"concept {concept_key!r} is a query meta-directive rather than an "
                f"answer fact ({canonical_meta_reason})"
            )
        canonical_weak_reason = variant_weak_reason(
            canonical,
            critical_values=critical_values,
            evidence_tokens=evidence_tokens,
            evidence_source_text=evidence_source_text,
            matcher=production_matcher,
        )
        if canonical_weak_reason is not None:
            weak_canonical_count += 1
            removed_variants.append(
                {
                    "concept_key": concept_key,
                    "variant": canonical,
                    "reason": f"weak_canonical:{canonical_weak_reason}",
                }
            )

        variants: list[str] = []
        seen_variants: set[str] = set()
        for variant_index, raw_variant in enumerate(
            [canonical, *(concept.get("any_of") or [])]
        ):
            variant = str(raw_variant or "").strip()
            normalized = policy_normalized(variant)
            if not normalized or normalized in seen_variants:
                continue
            weak_reason = variant_weak_reason(
                variant,
                critical_values=critical_values,
                evidence_tokens=evidence_tokens,
                evidence_source_text=evidence_source_text,
                matcher=production_matcher,
            )
            if weak_reason is not None:
                if variant_index == 0:
                    seen_variants.add(normalized)
                    continue
                removed_variants.append(
                    {
                        "concept_key": concept_key,
                        "variant": variant,
                        "reason": weak_reason,
                    }
                )
                seen_variants.add(normalized)
                continue
            seen_variants.add(normalized)
            variants.append(variant)

        normalized_key = policy_normalized(concept_key)
        if not normalized_key or not variants:
            raise ValueError(
                f"concept {concept_key!r} has no discriminative canonical/variant"
            )
        if normalized_key in concept_keys:
            raise ValueError(f"duplicate materialized concept_key {concept_key!r}")
        concept_keys.add(normalized_key)
        provisional_groups.append({"concept_key": concept_key, "any_of": variants})

    if policy_version == MATERIALIZATION_POLICY_VERSION:
        # v6把跨组完全相同的variant也交给同一个全局冲突图。预先从双方删除会
        # 破坏可行性，例如[g1: shared, alt]、[g2: shared]本可选择alt+shared。
        exact_filtered_groups = deepcopy(provisional_groups)
    else:
        # v5及更早版本的行为属于冻结审计事实，仍保留原有双边删除语义。
        owner_counts = Counter(
            normalized
            for group in provisional_groups
            for normalized in {
                policy_normalized(value) for value in group.get("any_of") or []
            }
        )
        exact_filtered_groups: list[dict[str, Any]] = []
        for group in provisional_groups:
            concept_key = str(group["concept_key"])
            variants: list[str] = []
            for variant in group.get("any_of") or []:
                normalized = policy_normalized(variant)
                if owner_counts[normalized] > 1:
                    removed_variants.append(
                        {
                            "concept_key": concept_key,
                            "variant": str(variant),
                            "reason": "cross_concept_ambiguous_variant",
                        }
                    )
                    continue
                variants.append(str(variant))
            if not variants:
                raise ValueError(
                    f"concept {concept_key!r} has no unique discriminative canonical/variant"
                )
            exact_filtered_groups.append(
                {"concept_key": concept_key, "any_of": variants}
            )

    if policy_version == MATERIALIZATION_POLICY_VERSION:
        selected_groups, subset_audit = _maximum_collision_free_subsets(
            exact_filtered_groups, matcher=production_matcher
        )
        selected = {
            (index, policy_normalized(variant))
            for index, group in enumerate(selected_groups)
            for variant in group["any_of"]
        }
        for group_index, group in enumerate(exact_filtered_groups):
            for variant in group.get("any_of") or []:
                if (group_index, policy_normalized(variant)) in selected:
                    continue
                removed_variants.append(
                    {
                        "concept_key": str(group["concept_key"]),
                        "variant": str(variant),
                        "reason": "global_collision_free_subset_exclusion",
                    }
                )
        collision_count = sum(
            1
            for item in removed_variants
            if item["reason"] == "global_collision_free_subset_exclusion"
        )
        groups = selected_groups
    else:
        collisions = _collision_witnesses(
            exact_filtered_groups, matcher=production_matcher
        )
        groups = []
        signatures: set[tuple[str, ...]] = set()
        collision_count = 0
        for group_index, group in enumerate(exact_filtered_groups):
            concept_key = str(group["concept_key"])
            variants = []
            for variant in group.get("any_of") or []:
                normalized = policy_normalized(variant)
                witness = collisions.get((group_index, normalized))
                if witness is not None:
                    collision_count += 1
                    removed_variants.append(
                        {
                            "concept_key": concept_key,
                            "variant": str(variant),
                            "reason": "cross_concept_production_match_collision",
                            "witness_concept_key": witness[0],
                            "witness_variant": witness[1],
                        }
                    )
                    continue
                variants.append(str(variant))
            if not variants:
                raise ValueError(
                    f"concept {concept_key!r} has no production-discriminative variant"
                )
            signature = tuple(sorted(policy_normalized(value) for value in variants))
            if signature in signatures:
                raise ValueError("duplicate materialized concept group")
            signatures.add(signature)
            groups.append({"concept_key": concept_key, "any_of": variants})

    if not groups:
        raise ValueError("concept_groups.all_of would be empty")
    audit = {
        "policy_version": policy_version,
        "production_matcher": PRODUCTION_MATCHER,
        "removed_variant_count": len(removed_variants),
        "removed_variants": removed_variants,
        "weak_canonical_count": weak_canonical_count,
        "production_match_collision_count": collision_count,
    }
    if policy_version == MATERIALIZATION_POLICY_VERSION:
        audit["production_matcher_policy_version"] = production_matcher_policy
        audit["subset_selection"] = {
            **subset_audit,
            "selected_variant_count": sum(len(group["any_of"]) for group in groups),
        }
    elif policy_version == V5_MATERIALIZATION_POLICY_VERSION:
        audit["production_matcher_policy_version"] = production_matcher_policy
    return {"all_of": groups}, audit
