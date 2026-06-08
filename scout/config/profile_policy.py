"""Profile-driven preferences — engine classifies; profile judges."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ROLE_FAMILY_TIERS = ("target", "secondary", "stretch", "avoid")
ROLE_CENTRE_TIERS = ("target", "secondary", "stretch", "avoid")

SPECIALIST_DOMAIN_ALIASES: dict[str, str] = {
    "llm_training_platform": "ml_infrastructure",
    "block_storage_infrastructure": "storage_infrastructure",
    "filesystem_internals": "storage_infrastructure",
    "distributed_storage_systems": "storage_infrastructure",
    "big_data_connectors": "jvm_data_infrastructure",
    "data_framework_integrations": "jvm_data_infrastructure",
    "jvm_performance_engineering": "jvm_data_infrastructure",
    "database_driver_engineering": "jvm_data_infrastructure",
}
SKILL_TIERS = (
    "strongest",
    "strong_recent",
    "professional",
    "stretch",
    "avoid_overclaiming",
    "missing_or_avoid_overclaiming",
)


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _skill_patterns(skill: str) -> list[re.Pattern[str]]:
    key = _norm_skill(skill)
    aliases: dict[str, list[str]] = {
        "php": [r"\bphp\b", r"\blaravel\b", r"\bmagento\b", r"\badobe commerce\b"],
        "adobe commerce": [r"\badobe commerce\b", r"\bmagento\b"],
        "backend apis": [r"\b(rest|graphql|api design|microservice)\b"],
        "postgresql": [r"\bpostgres(?:ql)?\b"],
        "mysql": [r"\bmysql\b"],
        "docker": [r"\bdocker\b"],
        "ci/cd": [r"\bci/?cd\b", r"\bgithub actions\b"],
        "opensearch": [r"\bopensearch\b", r"\belasticsearch\b"],
        "technical leadership": [r"\btechnical lead\b", r"\bengineering lead\b"],
        "typescript": [r"\btypescript\b"],
        "node.js": [r"\bnode\.?js\b"],
        "python": [r"\bpython\b"],
        "fastapi": [r"\bfastapi\b"],
        "llm workflows": [r"\b(llm|litellm|rag|agentic)\b"],
        "litellm": [r"\blitellm\b"],
        "event-driven workers": [r"\b(event[- ]driven|pub/sub|async worker)\b"],
        "react": [r"\breact\b"],
        "aws": [r"\baws\b"],
        "gcp": [r"\bgcp\b|\bgoogle cloud\b"],
        "redis": [r"\bredis\b"],
        "go": [r"\b(?:go|golang)\b"],
        "kubernetes": [r"\b(kubernetes|k8s|\beks\b)\b"],
        "django": [r"\bdjango\b"],
        "deep postgresql internals": [r"\b(postgres(?:ql)? guru|database internals)\b"],
        "jvm": [r"\bjvm\b"],
        "java": [r"\bjava\b", r"\bspring boot\b"],
        "rust": [r"\brust\b"],
        "elixir": [r"\belixir\b"],
        "ml research": [r"\bml research\b", r"\bresearch scientist\b"],
        "deep sre ownership": [r"\b(sre|site reliability).*(?:primary|owner)\b"],
    }
    pats = aliases.get(key, [rf"\b{re.escape(key)}\b"])
    return [re.compile(p, re.I) for p in pats]


@dataclass
class ScoringAdjustment:
    label: str
    delta: int
    source: str  # profile_preference | generic_blocker | cv_evidence_gap
    reason: str = ""


@dataclass
class ProfilePolicy:
    """Resolved active-profile preferences for scoring and filters."""

    raw: dict[str, Any]
    role_family_preferences: dict[str, list[str]] = field(default_factory=dict)
    skill_confidence: dict[str, list[str]] = field(default_factory=dict)
    location_policy: dict[str, Any] = field(default_factory=dict)
    contract_policy: dict[str, Any] = field(default_factory=dict)
    compensation_policy: dict[str, Any] = field(default_factory=dict)
    profile_penalties: dict[str, int] = field(default_factory=dict)
    profile_boosts: dict[str, int] = field(default_factory=dict)
    penalties: dict[str, int] = field(default_factory=dict)
    boosts: dict[str, int] = field(default_factory=dict)
    profile_name: str = "default"
    role_centre_preferences: dict[str, list[str]] = field(default_factory=dict)
    specialist_domain_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile_score_caps: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> ProfilePolicy:
        derived = _derive_policy_sections(profile)
        penalties = {**(profile.get("penalties") or {}), **derived.get("profile_penalties", {})}
        boosts = {**(profile.get("boosts") or {}), **derived.get("profile_boosts", {})}
        return cls(
            raw=profile,
            role_family_preferences=derived["role_family_preferences"],
            role_centre_preferences=derived["role_centre_preferences"],
            specialist_domain_preferences=derived["specialist_domain_preferences"],
            profile_score_caps=derived.get("profile_score_caps", {}),
            skill_confidence=derived["skill_confidence"],
            location_policy=derived["location_policy"],
            contract_policy=derived["contract_policy"],
            compensation_policy=derived["compensation_policy"],
            profile_penalties=derived.get("profile_penalties", {}),
            profile_boosts=derived.get("profile_boosts", {}),
            penalties=penalties,
            boosts=boosts,
            profile_name=str(profile.get("profile_name") or profile.get("candidate", {}).get("name", "default")),
        )

    def role_family_tier(self, family: str) -> str | None:
        prefs = self.role_family_preferences
        for tier in ROLE_FAMILY_TIERS:
            if family in (prefs.get(tier) or []):
                return tier
        return None

    def role_centre_tier(self, centre: str) -> str | None:
        prefs = self.role_centre_preferences
        for tier in ROLE_CENTRE_TIERS:
            if centre in (prefs.get(tier) or []):
                return tier
        if not prefs:
            return self.role_family_tier(centre)
        return None

    def specialist_domain_policy(self, domain: str) -> dict[str, Any] | None:
        sdp = self.specialist_domain_preferences
        if domain in sdp:
            return sdp[domain]
        alt = SPECIALIST_DOMAIN_ALIASES.get(domain)
        if alt and alt in sdp:
            return sdp[alt]
        return None

    def profile_score_cap(self, key: str) -> int | None:
        val = self.profile_score_caps.get(key)
        return int(val) if val is not None else None

    def classify_skill(self, skill: str, *, cv_text: str = "", jd_text: str = "") -> str:
        """Map a skill label to confidence tier for this profile."""
        skill_n = _norm_skill(skill)
        for tier in SKILL_TIERS:
            for entry in self.skill_confidence.get(tier) or []:
                if _norm_skill(entry) == skill_n:
                    return tier
        for tier in SKILL_TIERS:
            for entry in self.skill_confidence.get(tier) or []:
                if _norm_skill(entry) != skill_n:
                    continue
                for pat in _skill_patterns(entry):
                    if cv_text and pat.search(cv_text):
                        return tier
                    if jd_text and pat.search(jd_text) and not cv_text:
                        return tier
        if cv_text or jd_text:
            for pat in _skill_patterns(skill):
                if cv_text and pat.search(cv_text):
                    return "unknown"
                if jd_text and pat.search(jd_text) and not cv_text:
                    return "unknown"
        return "unknown"

    def skill_in_jd(self, skill: str, jd_text: str) -> bool:
        for pat in _skill_patterns(skill):
            if pat.search(jd_text):
                return True
        return False

    def candidate_location_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for c in self.location_policy.get("candidate_countries") or []:
            if c:
                tokens.add(_norm_skill(str(c)))
        loc = self.location_policy.get("candidate_location") or self.raw.get("candidate", {}).get(
            "location", ""
        )
        if loc:
            tokens.add(_norm_skill(loc))
        for city in self.location_policy.get("candidate_city_tokens") or []:
            if city:
                tokens.add(_norm_skill(str(city)))
        if "romania" in tokens:
            tokens.update({"romania", "bucharest"})
        tz = self.location_policy.get("timezone") or self.raw.get("candidate", {}).get("timezone", "")
        if tz:
            tokens.add(_norm_skill(tz.split("/")[-1]))
        return tokens

    def candidate_country_in_list(self, countries: set[str]) -> bool:
        tokens = self.candidate_location_tokens()
        for c in countries:
            cn = _norm_skill(c)
            if cn in tokens:
                return True
            for t in tokens:
                if t in cn or cn in t:
                    return True
        return False

    def list_must_include_candidate(self, countries: set[str], text: str) -> bool:
        """True if explicit country list is compatible with candidate location policy."""
        if self.candidate_country_in_list(countries):
            return True
        allow = self.location_policy.get("allowed_regions") or []
        text_l = text.lower()
        for region in allow:
            if region.lower() in text_l:
                return True
        explicit = self.location_policy.get("explicit_country_list_must_include") or []
        for token in explicit:
            if token.lower() in text_l or token.lower() in {c.lower() for c in countries}:
                return True
        return False

    def targets_devrel(self) -> bool:
        return self.role_family_tier("devrel_content") == "target"

    def avoids_devops_sre(self) -> bool:
        return self.role_family_tier("devops_sre_primary") == "avoid"

    def targets_frontend(self) -> bool:
        return self.role_family_tier("frontend_engineering") in ("target", "secondary")


def _derive_role_centre_preferences_from_families(
    rfp: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Map legacy role_family_preferences to role_centre_preferences when absent."""
    family_to_centre = {
        "backend_engineering": "backend_product_engineering",
        "platform_engineering": "backend_product_engineering",
        "ai_platform": "ai_platform",
        "product_engineering": "backend_product_engineering",
        "technical_lead": "engineering_lead",
        "fullstack_engineering": "fullstack_frontend_heavy",
        "data_engineering": "data_engineering",
        "infrastructure_engineering": "backend_product_engineering",
        "ml_engineering": "ml_infrastructure",
        "ml_infrastructure": "ml_infrastructure",
        "llm_training_platform": "llm_training_platform",
        "storage_infrastructure": "storage_infrastructure",
        "distributed_storage_systems": "storage_infrastructure",
        "block_storage_infrastructure": "storage_infrastructure",
        "filesystem_internals": "storage_infrastructure",
        "devops_sre_primary": "devops_sre_primary",
        "frontend_engineering": "frontend_product",
        "devrel_content": "devrel_content",
        "engineering_management": "engineering_management",
    }

    def _map_tier(tier: str) -> list[str]:
        centres: list[str] = []
        for family in rfp.get(tier) or []:
            centre = family_to_centre.get(family, family)
            if centre not in centres:
                centres.append(centre)
        return centres

    return {
        "target": _map_tier("target"),
        "secondary": _map_tier("secondary"),
        "stretch": _map_tier("stretch"),
        "avoid": _map_tier("avoid"),
    }


def _derive_policy_sections(profile: dict[str, Any]) -> dict[str, Any]:
    """Fill profile policy sections from legacy keys when new sections are absent."""
    out: dict[str, Any] = {}
    warned = False

    rfp = profile.get("role_family_preferences")
    if not rfp:
        rfp = {
            "target": [
                "backend_engineering",
                "platform_engineering",
                "ai_platform",
                "product_engineering",
                "technical_lead",
            ],
            "secondary": ["fullstack_engineering", "technical_lead", "data_engineering"],
            "stretch": ["infrastructure_engineering", "ml_engineering"],
            "avoid": [
                "devops_sre_primary",
                "frontend_engineering",
                "devrel_content",
                "sales",
                "engineering_management",
                "network_engineering",
                "data_operations",
            ],
        }
        warned = True
    out["role_family_preferences"] = rfp

    rcp = profile.get("role_centre_preferences")
    if not rcp:
        rcp = _derive_role_centre_preferences_from_families(rfp)
        warned = True
    out["role_centre_preferences"] = rcp

    sdp = profile.get("specialist_domain_preferences")
    if sdp is None:
        sdp = {}
        warned = True
    out["specialist_domain_preferences"] = sdp

    # `score_caps` = base-score band ceilings (practical_scoring); `profile_score_caps` = depth/specialist caps.
    base_cap_keys = {
        "base_below_45_max_final",
        "base_45_to_54_max_final",
        "base_55_to_64_max_final",
        "base_65_to_74_max_final",
        "base_75_plus_max_final",
    }
    merged_caps = dict(profile.get("profile_score_caps") or {})
    for key, val in (profile.get("score_caps") or {}).items():
        if key not in base_cap_keys and isinstance(val, (int, float)):
            merged_caps[key] = int(val)
    out["profile_score_caps"] = merged_caps

    sc = profile.get("skill_confidence")
    if not sc:
        sc = {
            "strongest": ["Python", "PostgreSQL", "Docker"],
            "strong_recent": ["TypeScript", "Kubernetes", "FastAPI"],
            "professional": ["Redis", "AWS", "CI/CD"],
            "stretch": ["Go", "GraphQL"],
            "avoid_overclaiming": ["Ceph", "block storage", "filesystem internals"],
        }
        warned = True
    out["skill_confidence"] = sc

    lp = profile.get("location_policy")
    cand = profile.get("candidate") or {}
    pipeline = profile.get("pipeline") or {}
    if not lp:
        cand_loc = (cand.get("location") or "").strip()
        cand_norm = _norm_skill(cand_loc)
        legacy_geo_on = pipeline.get("reject_country_restricted_remote", True)
        is_eu_remote_candidate = cand_norm in (
            "romania",
            "bulgaria",
            "poland",
            "ukraine",
            "hungary",
            "czech republic",
        ) or (
            "europe" in cand_norm
            and cand_norm not in {"united states", "usa", "us", "u.s.", "canada"}
        )
        eu_remote_default = legacy_geo_on and is_eu_remote_candidate
        if eu_remote_default:
            lp = {
                "candidate_location": cand_loc,
                "timezone": cand.get("timezone", ""),
                "allowed_broad_regions": [
                    "Europe",
                    "EU",
                    "EMEA",
                    "UK",
                    "UK/EU",
                    "Global",
                    "Worldwide",
                ],
                "blocked_regions": [
                    "US-only",
                    "LATAM-only",
                    "India-only",
                    "APAC-only",
                    "UK-only",
                ],
                "reject": {
                    "country_only": [
                        "us_only",
                        "ca_only",
                        "india_only",
                        "apac_only",
                        "uk_only",
                        "latam_only",
                    ],
                    "us_hub_without_broad_remote": True,
                    "timezone_locked": ["pacific", "us_eastern", "apac"],
                    "onsite_unless_candidate_mentioned": True,
                    "hybrid_unless_candidate_mentioned": True,
                    "single_country_list_excludes_candidate": True,
                },
                "explicit_country_list_must_include": [
                    "Europe",
                    "EU",
                    "EMEA",
                    "Global",
                    "Worldwide",
                ],
            }
        else:
            lp = {
                "candidate_location": cand_loc,
                "timezone": cand.get("timezone", ""),
                "geo_filter_mode": "off"
                if not pipeline.get("reject_country_restricted_remote", False)
                else "permissive",
                "allowed_broad_regions": [],
                "reject": {
                    "country_only": [],
                    "us_hub_without_broad_remote": False,
                    "timezone_locked": [],
                    "onsite_unless_candidate_mentioned": False,
                    "hybrid_unless_candidate_mentioned": False,
                    "single_country_list_excludes_candidate": False,
                },
            }
        warned = True
    out["location_policy"] = lp

    cp = profile.get("contract_policy") or {}
    legacy_pipeline = profile.get("pipeline") or {}
    if not cp.get("minimum_contract_months"):
        cp = {
            **cp,
            "permanent_preferred": cp.get("permanent_preferred", True),
            "allow_b2b_long_term": cp.get("allow_b2b_long_term", True),
            "allow_contract_unknown": cp.get("allow_contract_unknown", True),
            "reject_short_contracts": cp.get(
                "reject_short_contract_roles", legacy_pipeline.get("reject_contract_roles", True)
            ),
            "minimum_contract_months": cp.get("minimum_contract_months", 12),
            "reject_inside_ir35": cp.get("reject_inside_ir35", True),
            "reject_uk_payroll_only": cp.get("reject_uk_payroll_only", True),
        }
    out["contract_policy"] = cp

    comp = profile.get("compensation") or profile.get("compensation_policy") or {}
    out["compensation_policy"] = comp

    pp = profile.get("profile_penalties") or {}
    if not pp:
        pp = {
            "role_family_avoid": -20,
            "role_family_stretch": -8,
            "primary_skill_stretch": -8,
            "primary_skill_avoid_overclaiming": -15,
            "frontend_heavy_when_not_target": -8,
            "devops_sre_when_not_target": -15,
            "on_call_when_not_preferred": -8,
            "high_competition_without_exact_match": -8,
        }
    out["profile_penalties"] = pp

    pb = profile.get("profile_boosts") or {}
    if not pb:
        pb = {
            "target_role_family": 10,
            "strongest_skill_primary_match": 10,
            "strong_recent_skill_match": 7,
            "backend_platform_api_focus": 8,
            "ai_workflows_real_product": 10,
            "technical_lead_hands_on": 7,
            "remote_policy_match": 8,
        }
    out["profile_boosts"] = pb

    if warned:
        logger.debug(
            "Profile %s: derived role_family_preferences/skill_confidence/location_policy from legacy config",
            cand.get("name", "unknown"),
        )
    return out


def _skills_flat(skill_confidence: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for tier in SKILL_TIERS:
        for s in skill_confidence.get(tier) or []:
            out.add(_norm_skill(s))
    return out


def _combine_patterns(patterns: list[re.Pattern[str]]) -> re.Pattern[str]:
    parts = [p.pattern for p in patterns if p.pattern]
    if not parts:
        return re.compile(r"$^")
    return re.compile("|".join(f"(?:{p})" for p in parts), re.I)


def profile_targets_regulated_roles(profile: dict[str, Any]) -> bool:
    """True if profile explicitly targets clinical/licensed roles."""
    policy = ProfilePolicy.from_profile(profile)
    return policy.role_family_tier("healthcare_clinical") in ("target", "secondary")


def merge_profiles(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overlay onto base (overlay wins for scalars and list replacement)."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_profiles(result[key], value)
        else:
            result[key] = value
    return result
