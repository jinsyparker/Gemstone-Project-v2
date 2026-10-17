"""Resumable, polite scraper for gempundit.com loose-stone photographs.

Stages (each is cached on disk and safe to re-run):
  discover enumerate candidate products from the public sitemap (class = URL stem;
           category page 1 only validates the mapping) -> dataset/products.csv
  sample   seeded, per-class SKU order        -> dataset/sample_order.csv
  product  fetch product pages (in sample order until the per-class image target is
           met) -> dataset/raw_meta/<SKU>.json
  images   download "stone" photos at original resolution -> dataset/raw/<slug>/<file>
           and write dataset/scraped_images.csv

Usage:  python code/scrape.py {discover|sample|product|images|all} [--classes ruby emerald] [--target N]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import class_by_slug, load_config, resolve  # noqa: E402

log = logging.getLogger("scrape")

PRODUCT_COLS = ["class", "slug", "sku", "stem", "product_url"]


# --------------------------------------------------------------------------- http
class Throttle:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class Fetcher:
    def __init__(self, cfg: dict):
        s = cfg["scrape"]
        self.cfg = s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = s["user_agent"]
        self.rp = robotparser.RobotFileParser()
        self._load_robots()
        page_delay = float(s["delay_page_sec"])
        crawl_delay = self.rp.crawl_delay(s["user_agent"]) or self.rp.crawl_delay("*")
        if crawl_delay and page_delay < crawl_delay:
            if s.get("permission_confirmed"):
                log.warning(
                    "robots.txt Crawl-delay is %ss; using %ss because permission_confirmed=true "
                    "in config.yaml. Keep the owner's written permission with DATASHEET.md.",
                    crawl_delay, page_delay)
            else:
                page_delay = float(crawl_delay)
                log.warning("permission_confirmed=false -> honouring robots Crawl-delay %ss", crawl_delay)
        self.throttles = {"page": Throttle(page_delay), "image": Throttle(float(s["delay_image_sec"]))}
        self.consecutive_403 = 0
        self.max_consecutive_403 = 5

    def _load_robots(self) -> None:
        url = self.cfg["base_url"] + "/robots.txt"
        r = self.session.get(url, timeout=self.cfg["timeout_sec"])
        r.raise_for_status()
        self.rp.parse(r.text.splitlines())

    def allowed(self, url: str) -> bool:
        if urlsplit(url).netloc.startswith("imgcdn"):
            return True  # separate CDN host without its own robots policy
        return self.rp.can_fetch(self.cfg["user_agent"], url)

    def get(self, url: str, kind: str = "page", stream: bool = False) -> requests.Response:
        """GET with throttle + retry. Returns the response for any non-retryable status
        (caller decides what a 404 means); aborts the whole run on 401/403."""
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(self.cfg["max_retries"] + 1):
            self.throttles[kind].wait()
            try:
                r = self.session.get(url, timeout=self.cfg["timeout_sec"], stream=stream)
            except requests.RequestException as e:
                wait = min(2 ** attempt * 2, 120)
                log.warning("%s -> %s; retry in %ss", url, e, wait)
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                # A lone 403 can be a per-URL rule (e.g. /page/N, pigeon-blood-ruby-*): report it to the
                # caller. Several in a row means we are really blocked: stop instead of hammering.
                self.consecutive_403 += 1
                if self.consecutive_403 >= self.max_consecutive_403:
                    raise RuntimeError(f"{self.consecutive_403} consecutive 401/403 (last: {url}): "
                                       "looks like we're blocked; stopping.")
                log.warning("HTTP %s for %s (consecutive: %d)", r.status_code, url, self.consecutive_403)
                return r
            self.consecutive_403 = 0
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 0)) or min(2 ** attempt * 2, 120)
                log.warning("%s -> HTTP %s; retry in %ss", url, r.status_code, wait)
                r.close()
                time.sleep(wait)
                continue
            return r
        raise RuntimeError(f"giving up on {url} after {self.cfg['max_retries']} retries")


# ------------------------------------------------------------------------ helpers
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, ensure_ascii=False).encode())


def strip_query(url: str) -> str:
    return url.split("?")[0].strip()


def image_role(filename: str, scfg: dict) -> str:
    """stone | hand | certificate | other, from tokens after 'gp<id>-' in the filename."""
    stem = filename.rsplit(".", 1)[0].lower()
    tokens = re.split(r"[-_]", stem)[1:]
    if any(t in scfg["cert_tokens"] for t in tokens):
        return "certificate"
    if any(t in scfg["hand_tokens"] for t in tokens):
        return "hand"
    if tokens and tokens[0].isdigit():
        return "stone"
    return "other"


def title_ok(title: str, klass: dict, scfg: dict) -> bool:
    """Title names the class and is a single loose stone (no pair/set/jewellery words)."""
    t = (title or "").lower()
    if not any(k in t for k in klass["keywords"]):
        return False
    words = set(re.findall(r"[a-z]+", t))
    return not words & set(scfg["exclude_title_words"])


# --------------------------------------------------------------------- discovery
SKU_RE = re.compile(r"^(?P<stem>.+?)-gp(?P<num>\d+)(?P<tail>[a-z]?)$", re.I)


def parse_listing(html: str) -> list[dict]:
    """Product cards on a category listing page (used only to validate stem->class)."""
    soup = BeautifulSoup(html, "lxml")
    items, seen = [], set()
    for li in soup.select("li.item[data-productsku]"):
        sku = li["data-productsku"].upper()
        a = li.select_one("a.product-image")
        if not sku or sku in seen or a is None:
            continue
        seen.add(sku)
        items.append({"sku": sku, "title": li.get("data-productname", "").strip(),
                      "ptype": a.get("data-ptype", ""), "product_url": a.get("href", "")})
    return items


def stage_discover(cfg: dict, fetcher: Fetcher) -> None:
    """Enumerate candidate products from the public sitemap (the /page/N listing URLs
    return HTTP 403 to our bot UA, and we do not try to evade that). Class = URL stem.
    Category page 1 of every class is fetched only to measure stem->class agreement."""
    html_dir, ds = resolve(cfg, "raw_html"), resolve(cfg, "dataset")
    base = cfg["scrape"]["base_url"]
    stem_to_class = {}
    for c in cfg["classes"]:
        for s in c["url_stems"]:
            assert s not in stem_to_class, f"stem {s!r} assigned to two classes"
            stem_to_class[s] = c
    # 1) sitemap index -> product sitemaps (cached)
    idx_cache = html_dir / "sitemap" / "sitemap_index.xml.gz"
    if not idx_cache.exists():
        r = fetcher.get("https://gempundit.com/sitemapxml/sitemap_index.xml")
        r.raise_for_status()
        atomic_write(idx_cache, gzip.compress(r.content))
    index_xml = gzip.decompress(idx_cache.read_bytes()).decode("utf8")
    sitemap_urls = [u for u in re.findall(r"<loc>([^<]+)</loc>", index_xml) if "/products" in u]
    rows, seen = [], set()
    for su in sitemap_urls:
        cache = html_dir / "sitemap" / (su.rsplit("/", 1)[-1] + ".gz")
        if not cache.exists():
            r = fetcher.get(su)
            r.raise_for_status()
            atomic_write(cache, gzip.compress(r.content))
            log.info("fetched %s", su)
        xml = gzip.decompress(cache.read_bytes()).decode("utf8")
        for url in re.findall(r"<loc>([^<]+)</loc>", xml):
            m = SKU_RE.match(url.rstrip("/").rsplit("/", 1)[-1])
            if not m or m["stem"] not in stem_to_class:
                continue
            sku = f"GP{m['num']}{m['tail'].upper()}"
            if sku in seen:
                continue
            seen.add(sku)
            k = stem_to_class[m["stem"]]
            rows.append({"class": k["name"], "slug": k["slug"], "sku": sku, "stem": m["stem"],
                         "product_url": url})
    with open(ds / "products.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRODUCT_COLS)
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r_ in rows:
        counts[r_["slug"]] = counts.get(r_["slug"], 0) + 1
    log.info("products.csv: %d candidate products %s", len(rows), counts)
    # 2) validation against category page 1
    sku_slug = {r_["sku"]: r_["slug"] for r_ in rows}
    check = []
    for c in cfg["classes"]:
        cache = html_dir / "listing" / c["slug"] / "page_1.html.gz"
        if not cache.exists():
            r = fetcher.get(f"{base}/gemstones/{c['category']}")
            r.raise_for_status()
            atomic_write(cache, gzip.compress(r.text.encode("utf8")))
        items = parse_listing(gzip.decompress(cache.read_bytes()).decode("utf8"))
        agree = sum(sku_slug.get(i["sku"]) == c["slug"] for i in items)
        wrong = [(i["sku"], sku_slug.get(i["sku"])) for i in items if sku_slug.get(i["sku"]) != c["slug"]]
        check.append({"slug": c["slug"], "page1_items": len(items), "agree": agree,
                      "ptypes": sorted({i["ptype"] for i in items}), "disagree_examples": wrong[:8]})
        log.info("validate %-16s page1=%d agree=%d", c["slug"], len(items), agree)
    write_json(ds / "discovery_check.json", {"candidates_per_class": counts, "page1_validation": check})


# ------------------------------------------------------------------------ sample
def stage_sample(cfg: dict) -> None:
    """Per-class seeded random SKU order over the sitemap candidates. (Titles are
    checked on the product page in the next stage.)"""
    import pandas as pd

    ds = resolve(cfg, "dataset")
    df = pd.read_csv(ds / "products.csv", dtype=str, keep_default_na=False)
    out = []
    for slug, g in df.groupby("slug", sort=False):
        skus = sorted(g["sku"])  # sorted first so the shuffle is independent of file order
        random.Random(f"{cfg['seed']}-{slug}").shuffle(skus)
        out += [{"slug": slug, "class": g["class"].iloc[0], "rank": i, "sku": s} for i, s in enumerate(skus)]
        log.info("sample %s: %d candidates", slug, len(skus))
    pd.DataFrame(out).to_csv(ds / "sample_order.csv", index=False)


# ----------------------------------------------------------------------- product
def parse_product(html: str, sku: str) -> dict:
    num = re.sub(r"\D", "", sku)
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.select_one("h1")
    pat = re.compile(
        r"https?:\\?/\\?/(imgcdn\d*\.gempundit\.com)\\?/media\\?/catalog\\?/product\\?/"
        r"((?!cache)[^\"'\s\\?)]*?\\?/gp" + num + r"-[^\"'\s\\?)]+?\.(?:jpe?g|png|webp))", re.I)
    images, seen = [], set()
    for m in pat.finditer(html):
        rel = m.group(2).replace("\\", "")
        fn = rel.rsplit("/", 1)[-1]
        if fn in seen:
            continue
        seen.add(fn)
        images.append({"filename": fn, "url": f"https://{m.group(1)}/media/catalog/product/{rel}",
                       "gallery_pos": len(images)})
    return {"sku": sku, "title": h1.get_text(strip=True) if h1 else "",
            "page_title": soup.title.get_text(strip=True) if soup.title else "", "images": images}


def meta_path(cfg: dict, sku: str) -> Path:
    return resolve(cfg, "meta") / f"{sku}.json"


def stage_product(cfg: dict, fetcher: Fetcher, slugs: list[str], target: int) -> None:
    import pandas as pd

    ds = resolve(cfg, "dataset")
    order = pd.read_csv(ds / "sample_order.csv", dtype=str)
    prods = pd.read_csv(ds / "products.csv", dtype=str, keep_default_na=False).set_index("sku")
    by_slug = class_by_slug(cfg)
    for slug in slugs:
        klass = by_slug[slug]
        skus = order[order["slug"] == slug].sort_values("rank", key=lambda s: s.astype(int))["sku"].tolist()
        n_imgs, n_prod = 0, 0
        for sku in skus:
            if n_imgs >= target:
                break
            mp = meta_path(cfg, sku)
            if mp.exists():
                meta = json.loads(mp.read_text())
            else:
                url = prods.loc[sku, "product_url"]
                r = fetcher.get(url)
                if r.status_code != 200:
                    log.warning("[%s] %s -> HTTP %s, skipping", slug, sku, r.status_code)
                    meta = {"sku": sku, "title": "", "images": [], "error": r.status_code}
                else:
                    meta = parse_product(r.text, sku)
                meta.update({"slug": slug, "class": klass["name"], "product_url": url,
                             "fetched_at": now()})
                write_json(mp, meta)
            meta["title_ok"] = title_ok(meta.get("title", ""), klass, cfg["scrape"])
            if meta["title_ok"]:
                n_imgs += sum(image_role(i["filename"], cfg["scrape"]) == "stone" for i in meta["images"])
                n_prod += 1
            if n_prod % 25 == 0:
                log.info("[%s] %d products, %d stone images", slug, n_prod, n_imgs)
        log.info("[%s] DONE: %d products, %d stone images (target %d, candidates %d)",
                 slug, n_prod, n_imgs, target, len(skus))


# ------------------------------------------------------------------------ images
def download_image(fetcher: Fetcher, url: str, dest: Path) -> dict:
    """Try the bare (original) URL, falling back to the 960px webp transform."""
    for variant, u in (("original", url), ("w960_webp", url + "?imgeng=/w_960/f_webp/cmpr_20/s_7")):
        r = fetcher.get(u, kind="image")
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and r.content:
            data = r.content
            actual_dest = dest if variant == "original" else dest.with_suffix(".webp")
            atomic_write(actual_dest, data)
            return {"status": 200, "variant": variant, "file": actual_dest.name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "scraped_at": now()}
        log.warning("%s (%s) -> HTTP %s", u, variant, r.status_code)
    return {"status": r.status_code, "variant": None, "file": None, "bytes": 0, "sha256": "",
            "scraped_at": now()}


def stage_images(cfg: dict, fetcher: Fetcher, slugs: list[str]) -> None:
    raw, ds = resolve(cfg, "raw"), resolve(cfg, "dataset")
    for slug in slugs:
        metas = [json.loads(p.read_text()) for p in sorted(resolve(cfg, "meta").glob("*.json"))]
        metas = [m for m in metas if m.get("slug") == slug and title_ok(m.get("title", ""), class_by_slug(cfg)[slug], cfg["scrape"])]
        todo = [(m, i) for m in metas for i in m["images"] if image_role(i["filename"], cfg["scrape"]) == "stone"]
        log.info("[%s] %d stone images across %d products", slug, len(todo), len(metas))
        for k, (m, img) in enumerate(todo, 1):
            dl = m.setdefault("downloads", {})
            dest = raw / slug / img["filename"]
            if img["filename"] in dl and dl[img["filename"]]["status"] == 200 and (raw / slug / dl[img["filename"]]["file"]).exists():
                continue
            dl[img["filename"]] = download_image(fetcher, img["url"], dest)
            write_json(meta_path(cfg, m["sku"]), m)
            if k % 50 == 0:
                log.info("[%s] %d/%d", slug, k, len(todo))
    write_scraped_csv(cfg)


def write_scraped_csv(cfg: dict) -> None:
    ds = resolve(cfg, "dataset")
    rows = []
    for p in sorted(resolve(cfg, "meta").glob("*.json")):
        m = json.loads(p.read_text())
        klass = class_by_slug(cfg).get(m.get("slug"))
        if not klass or not title_ok(m.get("title", ""), klass, cfg["scrape"]):
            continue
        for img in m["images"]:
            role = image_role(img["filename"], cfg["scrape"])
            d = m.get("downloads", {}).get(img["filename"])
            rows.append({
                "class": m["class"], "slug": m["slug"], "sku": m["sku"], "title": m["title"],
                "product_url": m["product_url"], "image_url": img["url"], "filename": img["filename"],
                "gallery_pos": img["gallery_pos"], "role": role,
                "local_path": f"dataset/raw/{m['slug']}/{d['file']}" if d and d["status"] == 200 else "",
                "http_status": d["status"] if d else "", "variant": d["variant"] if d else "",
                "bytes": d["bytes"] if d else "", "sha256": d["sha256"] if d else "",
                "scraped_at": d["scraped_at"] if d else "",
            })
    with open(ds / "scraped_images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("scraped_images.csv: %d rows (%d downloaded)", len(rows), sum(bool(r["local_path"]) for r in rows))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "sample", "product", "images", "all"])
    ap.add_argument("--classes", nargs="*", help="class slugs (default: all)")
    ap.add_argument("--target", type=int, help="override scrape.target_images_per_class")
    ap.add_argument("--config", help="path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = resolve(cfg, "dataset")
    ds.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ds / "scrape.log")])
    slugs = args.classes or [c["slug"] for c in cfg["classes"]]
    unknown = set(slugs) - set(class_by_slug(cfg))
    if unknown:
        ap.error(f"unknown class slugs: {sorted(unknown)}")
    target = args.target or cfg["scrape"]["target_images_per_class"]

    stages = ["discover", "sample", "product", "images"] if args.stage == "all" else [args.stage]
    fetcher = Fetcher(cfg) if set(stages) & {"discover", "product", "images"} else None
    for st in stages:
        log.info("=== stage: %s ===", st)
        if st == "discover":
            stage_discover(cfg, fetcher)
        elif st == "sample":
            stage_sample(cfg)
        elif st == "product":
            stage_product(cfg, fetcher, slugs, target)
        elif st == "images":
            stage_images(cfg, fetcher, slugs)


if __name__ == "__main__":
    main()
