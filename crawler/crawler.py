"""
Web Crawler - crawls a UCI website and saves pages as structured JSON.

Usage:
    python crawler/crawler.py https://catalogue.uci.edu/allcourses --type course_catalog
    python crawler/crawler.py https://policies.uci.edu --type policy
    python crawler/crawler.py https://policies.uci.edu --type policy --output data/raw/policies --delay 1.0 --max 300
"""

import argparse
import io
import json
import re
import time
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup


# ──────────────────────────────────────────────────────────────────────────────
# URL Utilities
# ──────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Remove fragments and query parameters."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/')
    return parsed._replace(fragment='', query='', path=path).geturl()


def url_to_filepath(base_dir: Path, url: str) -> Path:
    """Convert a URL to a local .json file path under base_dir."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/')

    if parsed.query:
        safe_query = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.query)
        path = f"{path}__{safe_query}"

    if parsed.fragment:
        safe_fragment = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.fragment)
        path = f"{path}__{safe_fragment}"

    if not path or path.endswith('/'):
        path += 'index'
    elif "." not in Path(path).name:
        path += '/index'
    else:
        path = str(Path(path).with_suffix(''))

    return base_dir / (path + ".json")


def is_allowed_url(url: str, origin: str) -> bool:
    """Return True only if url is on the same domain and under the same path prefix as origin."""
    parsed_url    = urllib.parse.urlparse(url)
    parsed_origin = urllib.parse.urlparse(origin)

    if parsed_url.netloc != parsed_origin.netloc:
        return False

    origin_prefix = parsed_origin.path.rstrip('/')
    return parsed_url.path.startswith(origin_prefix)


# ──────────────────────────────────────────────────────────────────────────────
# Course Catalog — HTML Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_courses(soup: BeautifulSoup, page_url: str) -> list | None:
    """
    Parse UCI-style course blocks from an HTML page.
    Returns a list of course dicts, or None if no course blocks are found.
    """
    blocks = soup.find_all('div', class_='courseblock')
    if not blocks:
        return None

    courses = []
    crawled_at = datetime.now(timezone.utc).isoformat()

    for block in blocks:
        course = {"url": page_url, "crawled_at": crawled_at}

        code_tag  = block.find('span', class_=re.compile(r'detail-code'))
        title_tag = block.find('span', class_=re.compile(r'detail-title'))
        hours_tag = block.find('span', class_=re.compile(r'detail-hours_html'))

        course["code"]  = code_tag.get_text(strip=True).rstrip('.')  if code_tag  else ""
        course["title"] = title_tag.get_text(strip=True).rstrip('.') if title_tag else ""

        course["units"] = ""
        if hours_tag:
            m = re.search(
                r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s+Units?",
                hours_tag.get_text(strip=True),
                re.IGNORECASE,
            )
            if m:
                course["units"] = m.group(1)

        desc_div = block.find('div', class_='courseblockextra')
        course["description"] = desc_div.get_text(" ", strip=True) if desc_div else ""

        metadata = {}
        for label_tag in block.find_all('span', class_='label'):
            label = label_tag.get_text(strip=True).rstrip(':').strip()
            parent = label_tag.parent
            if parent:
                label_tag.extract()
                value = parent.get_text(" ", strip=True)
                metadata[label] = value

        course["prerequisite"]   = metadata.get("Prerequisite", "")
        course["corequisite"]    = metadata.get("Corequisite", "")
        course["restrictions"]   = metadata.get("Restrictions", metadata.get("Restriction", ""))
        course["grading_option"] = metadata.get("Grading Option", "")
        course["repeatability"]  = metadata.get("Repeatability", "")

        known = {"Prerequisite", "Corequisite", "Restrictions", "Restriction",
                 "Grading Option", "Repeatability"}
        course["extra"] = {k: v for k, v in metadata.items() if k not in known}

        courses.append(course)

    return courses


# ──────────────────────────────────────────────────────────────────────────────
# Course Catalog — PDF Parser
# ──────────────────────────────────────────────────────────────────────────────

# UCI course header in PDFs: "DEPT 101A. Course Title. 4 Units."
_COURSE_HEADER_RE = re.compile(
    r"^([A-Z][A-Z &\/]*\d+[A-Z]?)\.\s+(.+?)\.\s+([\d]+(?:\.\d+)?(?:-[\d]+(?:\.\d+)?)?)\s+Units?\.",
    re.MULTILINE,
)
_PDF_METADATA_RE = re.compile(
    r"^(Prerequisite|Corequisite|Restriction(?:s)?|Grading Option|Repeatability"
    r"|Same as|Overlaps with|Concurrent with):\s*(.+)",
    re.IGNORECASE,
)


def parse_courses_from_pdf(pdf_bytes: bytes, url: str) -> list | None:
    """
    Extract UCI course listings from a PDF using the same output schema as parse_courses().
    Returns a list of course dicts or None if no course headers are detected.
    """
    try:
        pages_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
    except Exception as e:
        print(f"  [PDF error] {url}: {e}")
        return None

    full_text = "\n".join(pages_text)
    matches = list(_COURSE_HEADER_RE.finditer(full_text))
    if not matches:
        return None

    crawled_at = datetime.now(timezone.utc).isoformat()
    courses = []

    for i, m in enumerate(matches):
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[m.end():block_end].strip()

        course = {
            "url":            url,
            "crawled_at":     crawled_at,
            "code":           m.group(1).strip(),
            "title":          m.group(2).strip(),
            "units":          m.group(3).strip(),
            "description":    "",
            "prerequisite":   "",
            "corequisite":    "",
            "restrictions":   "",
            "grading_option": "",
            "repeatability":  "",
            "extra":          {},
        }

        desc_lines = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            meta_m = _PDF_METADATA_RE.match(line)
            if meta_m:
                key   = meta_m.group(1).lower()
                value = meta_m.group(2).strip()
                if "prerequisite" in key:
                    course["prerequisite"] = value
                elif "corequisite" in key:
                    course["corequisite"] = value
                elif "restriction" in key:
                    course["restrictions"] = value
                elif "grading" in key:
                    course["grading_option"] = value
                elif "repeatability" in key:
                    course["repeatability"] = value
                else:
                    course["extra"][meta_m.group(1)] = value
            else:
                desc_lines.append(line)

        course["description"] = " ".join(desc_lines)
        courses.append(course)

    return courses if courses else None


# ──────────────────────────────────────────────────────────────────────────────
# Policy — HTML Parser
# ──────────────────────────────────────────────────────────────────────────────

# Tags whose children we recurse into but don't extract text from directly
_CONTAINER_TAGS = {"div", "section", "article", "main", "aside", "nav",
                   "header", "footer", "ul", "ol", "dl", "blockquote"}


def extract_table_text(table) -> str:
    """
    Convert an HTML table into readable lines of text.
    Each row becomes one line with cells joined by ' | '.
    Header cells (<th>) and data cells (<td>) are treated identically.
    Empty rows are skipped.
    """
    lines = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        cells = [c for c in cells if c]  # drop empty cells
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _walk(elem, current_lines: list, sections: list, state: dict) -> None:
    """
    Recursively walk an element's direct children, collecting text into
    sections split by headings. Tables are handled atomically so their
    cells are never visited individually as descendants.
    """
    for child in elem.children:
        if not hasattr(child, "name") or child.name is None:
            continue

        if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = child.get_text(strip=True)
            if not text:
                continue
            # Flush accumulated content under the previous heading
            content = " ".join(current_lines).strip()
            if content:
                sections.append({
                    "heading": state["heading"],
                    "level":   state["level"],
                    "content": content,
                })
            current_lines.clear()
            state["heading"] = text
            state["level"]   = child.name

        elif child.name == "table":
            # Handle the entire table as one atomic unit — never descend into it
            text = extract_table_text(child)
            if text:
                current_lines.append(text)

        elif child.name in ("p", "li", "dt", "dd"):
            text = child.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

        elif child.name in _CONTAINER_TAGS:
            # Recurse into layout containers without extracting them directly
            _walk(child, current_lines, sections, state)


def extract_policy_sections(soup: BeautifulSoup, url: str) -> dict:
    """
    Extract a policy page's content organized by heading hierarchy.
    Each heading + its following content (paragraphs, lists, tables) becomes
    one section. Tables are converted to pipe-delimited text rows.
    """
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    sections    = []
    state       = {"heading": title or "Introduction", "level": "h1"}
    current_lines: list[str] = []

    body = soup.find("body") or soup
    _walk(body, current_lines, sections, state)

    # Flush the final section
    content = " ".join(current_lines).strip()
    if content:
        sections.append({
            "heading": state["heading"],
            "level":   state["level"],
            "content": content,
        })

    return {
        "type":       "policy_page",
        "url":        url,
        "title":      title,
        "sections":   sections,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Policy — PDF Parser
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes, url: str) -> dict | None:
    """
    Extract text from a policy PDF, chunked by page.
    Returns a policy_page dict or None on failure / empty PDF.
    """
    filename = url.rstrip("/").split("/")[-1]
    try:
        sections = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    sections.append({
                        "heading": f"Page {i}",
                        "level":   "page",
                        "content": " ".join(text.split()),
                    })
    except Exception as e:
        print(f"  [PDF error] {url}: {e}")
        return None

    if not sections:
        return None

    return {
        "type":       "policy_page",
        "url":        url,
        "title":      filename,
        "sections":   sections,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Page Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def extract_page_data(html: str, url: str, page_type: str) -> dict | None:
    """Route an HTML page to the correct parser based on crawl type."""
    soup = BeautifulSoup(html, "html.parser")

    if page_type == "course_catalog":
        courses = parse_courses(soup, url)
        if courses is None:
            return None  # Non-course HTML page on a course_catalog crawl — skip
        return {
            "type":       "course_page",
            "url":        url,
            "courses":    courses,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }

    if page_type == "policy":
        return extract_policy_sections(soup, url)

    return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Return all absolute HTTP/HTTPS links found in an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        abs_url = urllib.parse.urljoin(base_url, href)
        if urllib.parse.urlparse(abs_url).scheme in ("http", "https"):
            links.append(abs_url)
    return links


# ──────────────────────────────────────────────────────────────────────────────
# Crawler
# ──────────────────────────────────────────────────────────────────────────────

def crawl(
    start_url:   str,
    page_type:   str,
    base_dir:    Path,
    delay:       float = 0.5,
    max_pages:   int   = 500,
    timeout:     int   = 10,
    max_retries: int   = 4,
) -> None:
    start_url = normalize_url(start_url)
    base_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PyCrawler/1.0)"})

    visited:      set[str]       = set()
    queue:        deque[str]     = deque([start_url])
    retry_counts: dict[str, int] = {}

    print(f"Crawl type  : {page_type}")
    print(f"Starting URL: {start_url}")
    print(f"Output dir  : {base_dir.resolve()}")
    print(f"Delay: {delay}s  |  Max pages: {max_pages}")
    print()

    while queue and len(visited) < max_pages:
        url = normalize_url(queue.popleft())

        if url in visited:
            continue
        if not is_allowed_url(url, start_url):
            continue

        visited.add(url)

        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            data = None

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                if page_type == "course_catalog":
                    courses = parse_courses_from_pdf(response.content, url)
                    if courses:
                        data = {
                            "type":       "course_page",
                            "url":        url,
                            "courses":    courses,
                            "crawled_at": datetime.now(timezone.utc).isoformat(),
                        }
                        print(f"  [PDF] Found {len(courses)} course(s)")
                    else:
                        print(f"  [PDF-skip] No courses detected: {url}")
                elif page_type == "policy":
                    data = extract_pdf_text(response.content, url)
                    if data:
                        print(f"  [PDF] Extracted {len(data['sections'])} page(s)")

            elif "text/html" in content_type:
                data = extract_page_data(response.text, url, page_type)
                # Only HTML pages are used to discover new links
                for link in extract_links(response.text, url):
                    if link not in visited and is_allowed_url(link, start_url):
                        queue.append(link)

            else:
                print(f"  [skip] {content_type} — {url}")

            if data is not None:
                filepath = url_to_filepath(base_dir, url)
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[{len(visited)}/{max_pages}] {url}")

        except requests.RequestException as e:
            visited.discard(url)
            attempts = retry_counts.get(url, 0)
            if attempts < max_retries:
                retry_counts[url] = attempts + 1
                queue.append(url)
                print(f"[retry {attempts + 1}/{max_retries}] {url} — {e}")
            else:
                visited.add(url)
                print(f"[give up] {url} — {e}")

        time.sleep(delay)

    print(f"\nCrawl complete. Pages visited: {len(visited)}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crawl a UCI website and save pages as structured JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "url",
        help="Starting URL (e.g. https://catalogue.uci.edu/allcourses)",
    )
    parser.add_argument(
        "--type", choices=["course_catalog", "policy"], required=True,
        help=(
            "course_catalog: extracts UCI course blocks from HTML and PDFs. "
            "policy: extracts heading-sectioned text from HTML and PDFs."
        ),
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory. Defaults to data/raw/courses or data/raw/policies based on --type.",
    )
    parser.add_argument("--delay",   "-d", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument("--max",     "-m", type=int,   default=500, help="Maximum pages to crawl")
    parser.add_argument("--timeout", "-t", type=int,   default=10,  help="Request timeout in seconds")
    args = parser.parse_args()

    default_dirs = {
        "course_catalog": "data/raw/courses",
        "policy":         "data/raw/policies",
    }
    output_dir = args.output or default_dirs[args.type]

    crawl(
        start_url = args.url,
        page_type = args.type,
        base_dir  = Path(output_dir),
        delay     = args.delay,
        max_pages = args.max,
        timeout   = args.timeout,
    )


if __name__ == "__main__":
    main()
