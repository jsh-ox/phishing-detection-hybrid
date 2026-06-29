"""
URL Feature Extractor
=====================
Reproduces the 18 URL-string-derived features used by the PhiUSIIL-based
anomaly corpus (G2), computed from a raw URL string. Used to convert URLs
extracted from CEAS/TREC emails into the same feature space the anomaly
models were trained on (G3 joint evaluation).

IMPORTANT (methodology note): this is a reimplementation of the PhiUSIIL
feature definitions. Exact parity with the original feature-extraction code
cannot be guaranteed for definitionally ambiguous features (obfuscation,
"other special chars"). This is an inherent limitation of cross-dataset
feature engineering and is documented in the dissertation.
"""

import math
import re
from urllib.parse import urlparse

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SPECIAL_COUNTED = set("=?&")  # counted individually; "other special" excludes these


def _parse(url: str):
    u = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u  # urlparse needs a scheme to populate netloc
    try:
        return urlparse(u)
    except ValueError:
        # malformed URL (e.g. stray brackets read as broken IPv6): fall back to
        # a best-effort manual split so a single bad string never halts the run.
        rest = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", u)
        host = re.split(r"[/?#]", rest, 1)[0]
        scheme = u.split("://", 1)[0].lower()
        return _SimpleParsed(scheme=scheme, netloc=host)


class _SimpleParsed:
    """Minimal stand-in for urlparse result when parsing fails."""
    def __init__(self, scheme="", netloc=""):
        self.scheme = scheme
        self.netloc = netloc


def extract_url_features(url: str) -> dict:
    url = str(url)
    parsed = _parse(url)
    netloc = parsed.netloc
    # strip port and userinfo
    host = netloc.split("@")[-1].split(":")[0]

    url_len = len(url)
    domain = host
    domain_len = len(domain)

    # PhiUSIIL conventions (derived empirically by matching recorded features):
    # the original extraction trims one trailing character from the URL before
    # counting, then counts letters/digits/specials over the URL with the scheme
    # prefix and a leading "www." removed. Ratios use this trimmed URLLength.
    url_trim = url[:-1] if url_len > 0 else url       # drop one trailing char
    url_len_phi = len(url_trim)
    body = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url_trim)  # strip scheme
    body = re.sub(r"^www\.", "", body, flags=re.IGNORECASE)       # strip leading www.

    # IP domain
    is_ip = 1 if IP_RE.match(host) else 0

    # TLD + subdomain counting
    parts = host.split(".") if host else []
    tld = parts[-1] if len(parts) >= 1 else ""
    tld_len = len(tld)
    # subdomains = labels beyond domain.tld  (e.g. a.b.example.com -> 2)
    n_subdomain = max(0, len(parts) - 2) if len(parts) >= 2 else 0

    # character counts over the URL with scheme + leading www removed
    letters = sum(c.isalpha() for c in body)
    digits = sum(c.isdigit() for c in body)
    n_equals = body.count("=")
    n_qmark = body.count("?")
    n_amp = body.count("&")

    # obfuscation: percent-encoding and non-ASCII chars
    pct = len(re.findall(r"%[0-9a-fA-F]{2}", body))
    non_ascii = sum(ord(c) > 127 for c in body)
    n_obfuscated = pct + non_ascii
    has_obfuscation = 1 if n_obfuscated > 0 else 0
    obfuscation_ratio = n_obfuscated / url_len_phi if url_len_phi else 0.0

    # "other special" = non-alphanumeric, excluding the individually counted ones
    other_special = sum(
        (not c.isalnum()) and (c not in SPECIAL_COUNTED) for c in body
    )
    total_special = sum((not c.isalnum()) for c in body)

    is_https = 1 if parsed.scheme == "https" else 0

    return {
        "URLLength": url_len_phi,
        "DomainLength": domain_len,
        "IsDomainIP": is_ip,
        "TLDLength": tld_len,
        "NoOfSubDomain": n_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": n_obfuscated,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / url_len_phi if url_len_phi else 0.0,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / url_len_phi if url_len_phi else 0.0,
        "NoOfEqualsInURL": n_equals,
        "NoOfQMarkInURL": n_qmark,
        "NoOfAmpersandInURL": n_amp,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": total_special / url_len_phi if url_len_phi else 0.0,
        "IsHTTPS": is_https,
    }


# Canonical feature order (must match G2 feature_schema.json)
FEATURE_ORDER = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
    "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsHTTPS",
]
