"""Small helpers to write the report as HTML and print it to PDF with Edge/Chrome."""
import html as _html
import os, re, subprocess, pathlib

CSS = r"""
@page { size: A4; margin: 20mm 19mm 20mm 21mm;
  @bottom-center { content: counter(page); font: 9pt "Segoe UI", sans-serif; color: #55544f; }
  @top-right { content: "CE60258  |  33 m PSC I-girder bridge B33"; font: 8pt "Segoe UI", sans-serif; color: #8a8984; } }
@page :first { @bottom-center { content: none; } @top-right { content: none; } }
html { font-size: 10.4pt; }
body { font-family: Cambria, "Times New Roman", serif; color: #1d1d1b; line-height: 1.42; margin: 0; }
h1, h2, h3, h4, .cap, th, .small, .toc, .titlepage .meta { font-family: "Segoe UI", Calibri, sans-serif; }
h1 { font-size: 15.5pt; margin: 0 0 8pt; padding-top: 2pt; break-before: page; color: #102a43; }
h1.nobreak { break-before: auto; }
h2 { font-size: 12pt; margin: 14pt 0 5pt; color: #102a43; break-after: avoid; }
h3 { font-size: 10.6pt; margin: 10pt 0 3pt; color: #243b53; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 16pt; padding: 0; } li { margin-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 8.6pt; break-inside: avoid; }
th { background: #eef2f6; font-weight: 600; text-align: left; }
th, td { border: 0.5pt solid #b8c2cc; padding: 2.2pt 4pt; vertical-align: top; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.cap { font-size: 8.4pt; color: #334e68; margin: 2pt 0 9pt; text-align: left; }
.tcap { font-size: 8.4pt; color: #334e68; margin: 8pt 0 2pt; font-family: "Segoe UI", Calibri, sans-serif; font-weight: 600; break-after: avoid; }
figure { margin: 6pt 0 4pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
.eq { margin: 5pt 0 7pt 24pt; font-style: normal; display: flex; justify-content: space-between; break-inside: avoid; }
.eq span.no { color: #55544f; }
.box { border-left: 2.5pt solid #2a78d6; background: #f4f7fb; padding: 5pt 8pt; margin: 6pt 0 9pt; break-inside: avoid; }
.small { font-size: 8.4pt; color: #52514e; }
.toc { font-size: 9.6pt; }
.toc div { display: flex; } .toc div span.t { flex: 1; } .toc div span.dots { flex: 1; border-bottom: 0.6pt dotted #8a8984; margin: 0 4pt 3pt; }
.toc div.l2 { padding-left: 14pt; font-size: 9pt; color: #334e68; }
.titlepage { height: 245mm; display: flex; flex-direction: column; }
.titlepage .top { font-family: "Segoe UI", sans-serif; color: #334e68; font-size: 10pt; letter-spacing: 0.3pt; }
.titlepage h0 { display: block; font-family: "Segoe UI", sans-serif; font-size: 23pt; font-weight: 600; color: #102a43; margin-top: 46mm; line-height: 1.2; }
.titlepage .sub { font-size: 13pt; color: #334e68; margin-top: 6mm; font-family: "Segoe UI", sans-serif; }
.titlepage .meta { margin-top: auto; font-size: 10pt; color: #243b53; line-height: 1.6; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; }
.keep { break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.6pt; }
"""


class Report:
    def __init__(self):
        self.parts = []
        self.toc = []          # (level, number, title, anchor)
        self.fig_no = 0
        self.tab_no = 0
        self.eq_no = 0
        self.h = [0, 0, 0]
        self.refs = {}

    def raw(self, s):
        self.parts.append(s)

    def h1(self, title, number=None, nobreak=False, toc=True):
        a = f"s{len(self.toc)}"
        if toc:
            self.toc.append((1, number, title, a))
        num = f"{number}&nbsp;&nbsp;" if number else ""
        self.parts.append(f'<h1 id="{a}" class="{"nobreak" if nobreak else ""}">{num}{title}</h1>')

    def h2(self, title, number=None):
        a = f"s{len(self.toc)}"
        self.toc.append((2, number, title, a))
        num = f"{number}&nbsp;&nbsp;" if number else ""
        self.parts.append(f'<h2 id="{a}">{num}{title}</h2>')

    def h3(self, title):
        self.parts.append(f"<h3>{title}</h3>")

    def p(self, *paras):
        for s in paras:
            self.parts.append(f"<p>{s}</p>")

    def ul(self, items):
        self.parts.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

    def box(self, s):
        self.parts.append(f'<div class="box">{s}</div>')

    def fig(self, name, caption, width=100):
        self.fig_no += 1
        self.parts.append(f'<figure><img src="../out/figs/{name}.png" style="width:{width}%">'
                          f'<div class="cap"><b>Figure {self.fig_no}.</b> {caption}</div></figure>')
        self.refs[name] = self.fig_no
        return self.fig_no

    def eq(self, body, tag=None):
        self.eq_no += 1
        if tag:
            self.refs[tag] = self.eq_no
        self.parts.append(f'<div class="eq"><span>{body}</span><span class="no">({self.eq_no})</span></div>')
        return self.eq_no

    def table(self, header, rows, caption=None, num_cols=(), widths=None, tag=None, small=False):
        self.tab_no += 1
        if tag:
            self.refs[tag] = self.tab_no
        out = []
        if caption:
            out.append(f'<div class="tcap">Table {self.tab_no}. {caption}</div>')
        style = ' style="font-size:7.9pt"' if small else ""
        out.append(f"<table{style}>")
        if widths:
            out.append("<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>")
        out.append("<tr>" + "".join(f'<th class="{"n" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(header)) + "</tr>")
        for r in rows:
            out.append("<tr>" + "".join(f'<td class="{"n" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
        out.append("</table>")
        self.parts.append("".join(out))
        return self.tab_no

    def toc_html(self, pages=None):
        rows = []
        for lvl, num, title, a in self.toc:
            pg = pages.get(a, "") if pages else ""
            label = f"{num}&nbsp;&nbsp;{title}" if num else title
            rows.append(f'<div class="l{lvl}"><span>{label}</span><span class="dots"></span><span>{pg}</span></div>')
        return '<div class="toc">' + "".join(rows) + "</div>"

    def html(self, pages=None):
        body = "\n".join(self.parts).replace("<!--TOC-->", self.toc_html(pages))

        def ref(m):
            if m.group(2) not in self.refs:
                raise KeyError(f"unresolved reference {m.group(0)}")
            return str(self.refs[m.group(2)])
        body = re.sub(r"\[\[(fig|tab):([\w-]+)\]\]", ref, body)
        return f'<!doctype html><html><head><meta charset="utf-8"><title>B33 integrated study</title><style>{CSS}</style></head><body>{body}</body></html>'


def browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Edge or Chrome needed to print the PDF")


def print_pdf(html_path, pdf_path):
    html_path = pathlib.Path(html_path).resolve(); pdf_path = pathlib.Path(pdf_path).resolve()
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    return pdf_path


def heading_pages(pdf_path, rep):
    """Find the page of every heading in the printed PDF (for the contents list)."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    texts = [pg.get_text() for pg in doc]
    out = {}
    toc_idx = next((i for i, t in enumerate(texts) if "Contents" in t and "Appendix" in t), 1)
    start = 1
    for lvl, num, title, a in rep.toc:
        if title != "Summary":
            start = max(start, toc_idx + 1)
        key = re.sub(r"<[^>]+>", "", (f"{num}  {title}" if num else title)).replace("&nbsp;", " ")
        key2 = re.sub(r"\s+", " ", key).strip()
        for i in range(start, len(texts)):
            if key2[:38] in re.sub(r"\s+", " ", texts[i]):
                out[a] = i + 1
                start = i
                break
    return out
