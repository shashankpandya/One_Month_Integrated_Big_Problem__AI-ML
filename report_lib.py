"""Helpers to write the report as HTML and print it to PDF with Edge/Chrome (headless)."""
import os, re, struct, subprocess, pathlib, hashlib

CSS = r"""
@page { size: A4; margin: 24mm 22mm 22mm 25mm;
  @bottom-center { content: counter(page); font-family: "Times New Roman", serif; font-size: 11pt; } }
@page :first { @bottom-center { content: none; } }
html { font-size: 12pt; }
body { font-family: "Times New Roman", Times, serif; color: #000; line-height: 1.42; margin: 0; }
p { margin: 0 0 7pt; text-align: justify; }
h1 { font-size: 16pt; font-weight: bold; margin: 26pt 0 10pt; break-after: avoid; }
h1.newpage { break-before: page; margin-top: 0; }
h2 { font-size: 13.5pt; font-weight: bold; margin: 16pt 0 6pt; break-after: avoid; }
h3 { font-size: 12pt; font-weight: bold; font-style: italic; margin: 11pt 0 4pt; break-after: avoid; }
ul, ol { margin: 0 0 8pt 22pt; padding: 0; }
li { margin-bottom: 3pt; text-align: justify; }
table { border-collapse: collapse; margin: 4pt auto 12pt; font-size: 10.5pt; line-height: 1.3; }
table.full { width: 100%; }
table.sm { font-size: 10pt; }
th, td { border: 0.6pt solid #444; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #e9e9e9; font-weight: bold; }
td.n { text-align: right; white-space: nowrap; }
th.n { text-align: right; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
.tcap { font-size: 11pt; text-align: center; margin: 12pt 0 3pt; break-after: avoid; }
figure { margin: 10pt 0 6pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
.cap { font-size: 11pt; text-align: center; margin: 4pt 6pt 12pt; line-height: 1.3; }
.eq { display: flex; align-items: center; margin: 6pt 0 9pt; break-inside: avoid; }
.eq .body { flex: 1; text-align: center; }
.eq .no { width: 32pt; text-align: right; }
.eq img { vertical-align: middle; }
.toc { font-size: 12pt; line-height: 1.55; }
.toc div { display: flex; align-items: baseline; }
.toc div span.t { white-space: nowrap; overflow: hidden; }
.toc div span.dots { flex: 1; border-bottom: 1pt dotted #555; margin: 0 4pt 3pt; min-width: 12pt; }
.toc div.l2 { padding-left: 22pt; }
.toc div.l1 { font-weight: bold; margin-top: 4pt; }
.lof div { display: flex; align-items: flex-end; font-size: 11.5pt; line-height: 1.45; }
.lof div span.k { width: 62pt; flex: none; align-self: flex-start; }
.lof div span.t { }
.lof div span.dots { flex: 1; border-bottom: 1pt dotted #555; margin: 0 4pt 3pt; min-width: 12pt; }
.titlepage { height: 247mm; display: flex; flex-direction: column; align-items: center; text-align: center; }
.titlepage .course { font-size: 13pt; margin-top: 8mm; line-height: 1.5; }
.titlepage .title { font-size: 21pt; font-weight: bold; margin-top: 22mm; line-height: 1.3; }
.titlepage .sub { font-size: 13pt; margin-top: 7mm; line-height: 1.45; font-style: italic; }
.titlepage .by { font-size: 13pt; margin-top: 18mm; line-height: 1.7; }
.titlepage .inst { font-size: 13pt; margin-top: auto; line-height: 1.6; }
.keep { break-inside: avoid; }
.flow { font-size: 11pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 10.5pt; }
"""


def png_size_in(path, dpi=200):
    with open(path, "rb") as fh:
        head = fh.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w / dpi, h / dpi


class Report:
    def __init__(self, eq_dir="report/eq"):
        self.parts = []
        self.toc = []                 # (level, number, title, anchor)
        self.figs = []                # (number, caption)
        self.tabs = []
        self.fig_no = self.tab_no = self.eq_no = 0
        self.refs = {}
        self.eq_dir = pathlib.Path(eq_dir)
        self.eq_dir.mkdir(parents=True, exist_ok=True)

    def raw(self, s):
        self.parts.append(s)

    def h1(self, title, number=None, newpage=False, toc=True):
        a = f"s{len(self.toc)}"
        if toc:
            self.toc.append((1, number, title, a))
        num = f"{number}&ensp;" if number else ""
        self.parts.append(f'<h1 id="{a}" class="{"newpage" if newpage else ""}">{num}{title}</h1>')

    def h2(self, title, number=None):
        a = f"s{len(self.toc)}"
        self.toc.append((2, number, title, a))
        num = f"{number}&ensp;" if number else ""
        self.parts.append(f'<h2 id="{a}">{num}{title}</h2>')

    def h3(self, title):
        self.parts.append(f"<h3>{title}</h3>")

    def p(self, *paras):
        for s in paras:
            self.parts.append(f"<p>{s}</p>")

    def ul(self, items, ordered=False):
        tag = "ol" if ordered else "ul"
        self.parts.append(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")

    def fig(self, name, caption, scale=1.0):
        self.fig_no += 1
        path = f"out/figs/{name}.png"
        w, h = png_size_in(path)
        self.parts.append(f'<figure><img src="../{path}" style="width:{w*scale:.3f}in">'
                          f'<div class="cap"><b>Figure {self.fig_no}:</b> {caption}</div></figure>')
        self.refs[name] = self.fig_no
        self.figs.append((self.fig_no, caption))
        return self.fig_no

    def eq(self, tex, tag=None, size=13):
        """Display equation typeset by matplotlib mathtext (STIX, matches Times)."""
        self.eq_no += 1
        if tag:
            self.refs[tag] = self.eq_no
        svg = self._tex_svg(tex, size)
        self.parts.append(f'<div class="eq"><span class="no"></span><span class="body"><img src="../{svg}"></span>'
                          f'<span class="no">({self.eq_no})</span></div>')
        return self.eq_no

    def _tex_svg(self, tex, size):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        key = hashlib.md5(f"{tex}|{size}".encode()).hexdigest()[:12]
        path = self.eq_dir / f"eq_{key}.svg"
        if not path.exists():
            with plt.rc_context({"mathtext.fontset": "stix", "font.family": "STIXGeneral", "svg.fonttype": "path"}):
                fig = plt.figure(figsize=(0.01, 0.01))
                fig.text(0, 0, f"${tex}$", fontsize=size)
                fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=True)
                plt.close(fig)
        return path.as_posix()

    def table(self, header, rows, caption=None, num_cols=(), widths=None, tag=None, small=False, full=True):
        out = []
        if caption:                   # uncaptioned tables (e.g. the notation list) are not numbered
            self.tab_no += 1
            if tag:
                self.refs[tag] = self.tab_no
            out.append(f'<div class="tcap"><b>Table {self.tab_no}:</b> {caption}</div>')
            self.tabs.append((self.tab_no, caption))
        cls = " ".join(c for c in ("full" if full else "", "sm" if small else "") if c)
        out.append(f'<table class="{cls}">')
        if widths:
            out.append("<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>")
        out.append("<thead><tr>" + "".join(f'<th class="{"n" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(header)) + "</tr></thead><tbody>")
        for r in rows:
            out.append("<tr>" + "".join(f'<td class="{"n" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
        out.append("</tbody></table>")
        self.parts.append("".join(out))
        return self.tab_no

    # ------------------------------------------------------------- lists with page numbers
    def toc_html(self, pages=None):
        rows = []
        for lvl, num, title, a in self.toc:
            pg = pages.get(a, "") if pages else ""
            label = f"{num}&ensp;{title}" if num else title
            rows.append(f'<div class="l{lvl}"><span class="t">{label}</span><span class="dots"></span><span>{pg}</span></div>')
        return '<div class="toc">' + "".join(rows) + "</div>"

    def lof_html(self, items, kind, pages=None):
        rows = []
        for n, cap in items:
            short = re.sub(r"<[^>]+>", "", cap).split(". ")[0].split(": ")[0]
            if short.count("(") > short.count(")"):          # cut inside a bracket, e.g. "(boxes: quartiles ..."
                short = short[:short.rfind(" (")]
            pg = pages.get(f"{kind}{n}", "") if pages else ""
            rows.append(f'<div><span class="k">{kind.capitalize()} {n}</span><span class="t">{short}</span>'
                        f'<span class="dots"></span><span>{pg}</span></div>')
        return '<div class="lof">' + "".join(rows) + "</div>"

    def html(self, pages=None):
        body = "\n".join(self.parts)
        body = body.replace("<!--TOC-->", self.toc_html(pages))
        body = body.replace("<!--LOF-->", self.lof_html(self.figs, "figure", pages))
        body = body.replace("<!--LOT-->", self.lof_html(self.tabs, "table", pages))

        def ref(m):
            if m.group(2) not in self.refs:
                raise KeyError(f"unresolved reference {m.group(0)}")
            return str(self.refs[m.group(2)])
        body = re.sub(r"\[\[(fig|tab|eq):([\w-]+)\]\]", ref, body)
        return (f'<!doctype html><html><head><meta charset="utf-8"><title>B33 integrated study</title>'
                f"<style>{CSS}</style></head><body>{body}</body></html>")


def browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Edge or Chrome is needed to print the PDF")


def print_pdf(html_path, pdf_path):
    html_path = pathlib.Path(html_path).resolve(); pdf_path = pathlib.Path(pdf_path).resolve()
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
    return pdf_path


def find_pages(pdf_path, rep):
    """Page numbers of headings, figures and tables in the printed PDF."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    texts = [re.sub(r"\s+", " ", pg.get_text()) for pg in doc]
    out = {}
    lists_end = max([i for i, t in enumerate(texts) if "List of Tables" in t] + [1])
    start = 1
    for lvl, num, title, a in rep.toc:
        key = re.sub(r"<[^>]+>", "", (f"{num} {title}" if num else title)).replace("&ensp;", " ").replace("&amp;", "&")
        key = re.sub(r"&[a-z]+;", " ", key)
        key = re.sub(r"\s+", " ", key).strip()[:40]
        if title not in ("Abstract", "Acknowledgement", "Contents", "List of Figures", "List of Tables", "Notation"):
            start = max(start, lists_end + 1)
        for i in range(start, len(texts)):
            if key in texts[i]:
                out[a] = i + 1
                start = i
                break
    for kind, items in (("figure", rep.figs), ("table", rep.tabs)):
        for n, _ in items:
            tag = f"{kind.capitalize()} {n}:"
            for i in range(lists_end + 1, len(texts)):
                if tag in texts[i]:
                    out[f"{kind}{n}"] = i + 1
                    break
    return out
