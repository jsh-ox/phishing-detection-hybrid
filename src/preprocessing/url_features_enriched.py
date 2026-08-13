"""
Enriched URL Feature Extractor
==============================
Project: Hybrid ML for Phishing Detection (anomaly layer)

A SINGLE shared feature function computed identically wherever URLs are
processed (PhiUSIIL training, CEAS evaluation, any future corpus). Using one
function everywhere guarantees parity by construction - the previous ~95%
reproduction drift came from having two separate implementations.

DESIGN CONSTRAINT: every feature is computable from the RAW URL STRING ALONE.
No external lookups (WHOIS, PageRank, reputation) - those are unavailable at
fusion time when only a URL extracted from an email is known.

Feature groups:
  - Character/entropy : randomness of the string (algorithmically-generated
                        domains, gibberish) - Shannon entropy, vowel ratios, runs
  - Lexical/keyword   : suspicious tokens, brand-in-path, hyphens, digit ratios
  - Structural/encoding: IP host, port, '@', encoding density, shorteners,
                         punycode/homograph markers, length ratios

All features are numeric so they drop straight into the existing anomaly models.
"""

import math
import re
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "bank", "confirm",
    "signin", "password", "webscr", "ebayisapi", "paypal", "submit", "click",
]
COMMON_BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "facebook", "netflix",
    "ebay", "instagram", "whatsapp", "linkedin", "dropbox", "chase", "wellsfargo",
]
SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorturl.at", "rebrand.ly",
}
VOWELS = set("aeiou")


def _shannon_entropy(s: str) -> float:
    """Bits of entropy in the character distribution of s."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _longest_run(s: str, predicate) -> int:
    best = cur = 0
    for c in s:
        if predicate(c):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _char_class_transitions(s: str) -> int:
    """Count transitions between letter / digit / other - churn signals gibberish."""
    def cls(c):
        if c.isalpha():
            return "a"
        if c.isdigit():
            return "d"
        return "o"
    trans = 0
    prev = None
    for c in s:
        k = cls(c)
        if prev is not None and k != prev:
            trans += 1
        prev = k
    return trans


def extract_enriched_features(url: str) -> dict:
    """Return a dict of enriched, URL-string-only features for one URL."""
    url = str(url) if url is not None else ""
    # Parse defensively - many raw URLs lack a scheme, and some are malformed
    # (e.g. unbalanced brackets that urlparse rejects as invalid IPv6). A single
    # bad URL must not halt a 700k-row run, so fall back to empty parse on error.
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc.lower()
    except ValueError:
        parsed = urlparse("http://")
        host = ""
    # strip a leading 'www.' and any port for host-level features
    host_noport = host.split(":")[0]
    host_nowww = host_noport[4:] if host_noport.startswith("www.") else host_noport
    path = parsed.path or ""
    domain_body = host_nowww.split(".")[0] if host_nowww else ""

    letters = [c for c in url if c.isalpha()]
    n_letters = len(letters) or 1
    url_len = len(url) or 1

    feats = {
        # --- Character / entropy ---
        "url_entropy": _shannon_entropy(url),
        "domain_entropy": _shannon_entropy(host_nowww),
        "vowel_ratio": sum(c.lower() in VOWELS for c in letters) / n_letters,
        "consonant_run_max": _longest_run(host_nowww, lambda c: c.isalpha() and c.lower() not in VOWELS),
        "digit_run_max": _longest_run(url, str.isdigit),
        "char_class_transitions": _char_class_transitions(host_nowww),
        "char_class_transition_ratio": _char_class_transitions(host_nowww) / (len(host_nowww) or 1),

        # --- Lexical / keyword ---
        "suspicious_keyword_count": sum(url.lower().count(k) for k in SUSPICIOUS_KEYWORDS),
        "has_suspicious_keyword": int(any(k in url.lower() for k in SUSPICIOUS_KEYWORDS)),
        "brand_in_path_not_domain": int(any(b in path.lower() and b not in host_nowww for b in COMMON_BRANDS)),
        "hyphen_count_domain": host_nowww.count("-"),
        "digit_ratio_domain": sum(c.isdigit() for c in host_nowww) / (len(host_nowww) or 1),
        "subdomain_depth": max(host_nowww.count(".") - 1, 0),
        "path_depth": path.count("/"),

        # --- Structural / encoding ---
        "is_ip_host": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host_noport))),
        "has_port": int(":" in host and not host.startswith("http")),
        "has_at_symbol": int("@" in url),
        "percent_encoding_count": url.count("%"),
        "percent_encoding_ratio": url.count("%") / url_len,
        "is_shortener": int(host_nowww in SHORTENER_HOSTS),
        "has_punycode": int("xn--" in host_noport),
        "domain_to_url_len_ratio": len(host_nowww) / url_len,
        "num_dots": url.count("."),
        "has_https": int(parsed.scheme == "https"),
    }
    return feats


# Ordered feature name list (stable column order everywhere)
ENRICHED_FEATURE_NAMES = list(extract_enriched_features("http://example.com").keys())


if __name__ == "__main__":
    # quick self-demonstration
    for u in [
        "https://www.paypal.com",
        "http://paypal.verify-account.secure-login.tk/webscr?cmd=update",
        "http://192.168.1.1:8080/login",
        "https://bit.ly/3xYz",
        "http://xn--pypal-4ve.com/confirm",
    ]:
        f = extract_enriched_features(u)
        print(f"\n{u}")
        print(f"  entropy={f['url_entropy']:.2f} susp_kw={f['suspicious_keyword_count']} "
              f"ip={f['is_ip_host']} shortener={f['is_shortener']} puny={f['has_punycode']} "
              f"brand_in_path={f['brand_in_path_not_domain']}")
