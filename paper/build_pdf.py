"""Typeset paper/numgate.tex as a PDF with ReportLab, without a LaTeX installation.

Usage: python paper/build_pdf.py   (run paper/make_numbers.py first)
Writes paper/numgate.pdf.

It reads the same LaTeX source a TeX engine would, and understands the subset the paper uses:
sections, run-in paragraph headings, itemize/enumerate/description lists, figures, the generated
results table, inline math, citations (natbib \\citep/\\citet), cross-references and the bibliography.
Numbers come from paper/numbers.tex, so the PDF and the .tex always agree.
"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAPER = Path(__file__).resolve().parent
FONT_DIR = "/System/Library/Fonts"
INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#444444")

# ---------------------------------------------------------------- fonts


def register_fonts():
    """Charter for text and Menlo for code (both ship with macOS); fall back to the built-in Times if missing."""
    try:
        charter = f"{FONT_DIR}/Supplemental/Charter.ttc"
        for name, idx in [("Serif", 0), ("Serif-Italic", 1), ("Serif-BoldItalic", 2), ("Serif-Bold", 3)]:
            pdfmetrics.registerFont(TTFont(name, charter, subfontIndex=idx))
        pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-BoldItalic")
        for name, idx in [("Mono", 0), ("Mono-Bold", 1)]:
            pdfmetrics.registerFont(TTFont(name, f"{FONT_DIR}/Menlo.ttc", subfontIndex=idx))
        return "Serif", "Mono"
    except Exception:
        return "Times-Roman", "Courier"


SERIF, MONO = register_fonts()


def style(name, **kw):
    base = dict(fontName=SERIF, fontSize=10.5, leading=14.2, textColor=INK, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "title": style("title", fontSize=17, leading=21, alignment=TA_CENTER, fontName=SERIF + "-Bold" if SERIF == "Serif" else "Times-Bold"),
    "author": style("author", fontSize=10.5, leading=13.5, alignment=TA_CENTER),
    "date": style("date", fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=MUTED),
    "abs_head": style("abs_head", fontSize=10, alignment=TA_CENTER, fontName="Serif-Bold" if SERIF == "Serif" else "Times-Bold"),
    "abstract": style("abstract", fontSize=9.6, leading=12.8, leftIndent=0.45 * inch, rightIndent=0.45 * inch),
    "h1": style("h1", fontSize=12.5, leading=16, alignment=TA_LEFT, spaceBefore=10, spaceAfter=4,
                fontName="Serif-Bold" if SERIF == "Serif" else "Times-Bold"),
    "body": style("body", spaceAfter=6),
    "item": style("item", spaceAfter=2),
    "caption": style("caption", fontSize=9, leading=11.5, alignment=TA_LEFT, textColor=MUTED),
    "cell": style("cell", fontSize=8.2, leading=10, alignment=TA_LEFT),
    "cellr": style("cellr", fontSize=8.2, leading=10, alignment=2),
    "ref": style("ref", fontSize=9, leading=11.6, alignment=TA_LEFT, leftIndent=18, firstLineIndent=-18, spaceAfter=3),
}

# ---------------------------------------------------------------- LaTeX helpers


def matching_brace(s, i):
    """Index of the brace closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced braces near: " + s[i:i + 60])


def arg(s, name):
    """Content of the first \\name{...} in s, with nested braces."""
    i = s.find("\\" + name + "{")
    if i < 0:
        return None
    start = i + len(name) + 1
    return s[start + 1:matching_brace(s, start)]


def load_macros():
    """\\newcommand definitions from numbers.tex (generated) as {name: value}."""
    macros = {}
    text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", text):
        start = m.end() - 1
        macros[m.group(1)] = text[start + 1:matching_brace(text, start)]
    return macros


def expand_macros(s, macros):
    for name in sorted(macros, key=len, reverse=True):
        s = re.sub(r"\\" + name + r"(\{\})?(?![A-Za-z])", lambda _m, v=macros[name]: v, s)
    return s


def parse_bib():
    """{key: fields} from references.bib (the simple one-field-per-line format used there)."""
    entries = {}
    text = (PAPER / "references.bib").read_text(encoding="utf-8")
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*$", m.group(3), flags=re.M)}
        fields["type"] = m.group(1).lower()
        entries[m.group(2).strip()] = fields
    return entries


def authors_of(entry):
    return [a.strip() for a in re.sub(r"[{}]", "", entry.get("author", "")).split(" and ")]


def surname(author):
    return author.split(",")[0].strip() if "," in author else author.split()[-1]


def cite_name(entry):
    names = [surname(a) for a in authors_of(entry) if a != "others"]
    if len(names) == 1:
        return names[0]
    if len(names) == 2 and "others" not in entry.get("author", ""):
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


class Refs:
    """Numbers citations like natbib with plainnat: entries sorted by first author's surname, then year."""

    def __init__(self, bib, body):
        cited = []
        for m in re.finditer(r"\\cite[pt]\{([^}]*)\}", body):
            for key in m.group(1).split(","):
                if key.strip() not in cited:
                    cited.append(key.strip())
        self.bib = bib
        self.keys = sorted(cited, key=lambda k: (surname(authors_of(bib[k])[0]).lower(), bib[k].get("year", "")))
        self.number = {k: i + 1 for i, k in enumerate(self.keys)}

    def citep(self, keys):
        nums = sorted(self.number[k.strip()] for k in keys.split(","))
        return "[" + ", ".join(str(n) for n in nums) + "]"

    def citet(self, keys):
        return "; ".join(f"{cite_name(self.bib[k.strip()])} [{self.number[k.strip()]}]" for k in keys.split(","))


def math_to_markup(expr):
    """The few inline formulas in the paper, as italic text with real symbols."""
    rep = {r"\le": "≤", r"\ge": "≥", r"\cdot": "·", r"\times": "×", r"\approx": "≈", r"\,": " "}
    for a, b in rep.items():
        expr = expr.replace(a, b)
    expr = re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", expr)
    expr = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", expr)
    expr = re.sub(r"\^(\w)", r"<super>\1</super>", expr)
    return "<i>" + expr + "</i>"


def inline(s, ctx):
    """LaTeX inline markup to ReportLab paragraph markup."""
    s = re.sub(r"(?<!\\)%.*", "", s)  # comments
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace(r"\&amp;", "&amp;")
    s = s.replace(r"\$", "\x00DOLLAR\x00")  # an escaped dollar is money, not the start of math
    s = re.sub(r"\$([^$]+)\$", lambda m: math_to_markup(m.group(1)), s)
    s = s.replace("\x00DOLLAR\x00", "$")
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: ctx["labels"].get(m.group(1), "??"), s)
    s = re.sub(r"\\citep\{([^}]*)\}", lambda m: ctx["refs"].citep(m.group(1)), s)
    s = re.sub(r"\\citet\{([^}]*)\}", lambda m: ctx["refs"].citet(m.group(1)), s)
    s = re.sub(r"\\url\{([^}]*)\}", r'<font name="%s" size="8.5">\1</font>' % MONO, s)
    # innermost-first rewrite of simple commands with one argument
    simple = {"textbf": ("<b>", "</b>"), "emph": ("<i>", "</i>"), "textit": ("<i>", "</i>"),
              "texttt": ('<font name="%s" size="8.8">' % MONO, "</font>"),
              "textcolor{red}": ('<font color="#c0262d">', "</font>")}
    changed = True
    while changed:
        changed = False
        for cmd, (a, b) in simple.items():
            new = re.sub(r"\\" + re.escape(cmd) + r"\{([^{}]*)\}", lambda m: a + m.group(1) + b, s)
            if new != s:
                s, changed = new, True
    for a, b in [(r"\pounds", "£"), (r"\%", "%"), (r"\$", "$"), (r"\_", "_"), (r"\#", "#"), ("``", "“"), ("''", "”"),
                 ("---", "—"), ("--", "–"), (r"\today", ctx["date"]), (r"\\", "<br/>"), ("\\ ", " ")]:
        s = s.replace(a, b)
    s = re.sub(r"(?<![\w])`", "‘", s).replace("'", "’")
    s = s.replace("~", "\u00a0")
    s = re.sub(r"\\(small|centering|hfill|noindent|par)\b\s*", "", s)
    s = re.sub(r"\{\}", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- document structure


def number_labels(body):
    """Section, table and figure numbers for \\ref, in order of appearance."""
    labels, sec, tab, fig = {}, 0, 0, 0
    env = None
    for m in re.finditer(r"\\section\{|\\begin\{(table|figure)\}|\\label\{([^}]*)\}", body):
        if m.group(0).startswith("\\section"):
            sec += 1
            env = ("sec", sec)
        elif m.group(1) == "table":
            tab += 1
            env = ("tab", tab)
        elif m.group(1) == "figure":
            fig += 1
            env = ("fig", fig)
        elif m.group(2) and env:
            labels[m.group(2)] = str(env[1])
    return labels


def split_blocks(body):
    """Top-level blocks: environments, sections and paragraphs, in order."""
    blocks, i = [], 0
    pat = re.compile(r"\\begin\{(abstract|itemize|enumerate|description|table|figure)\}(\[[^\]]*\])?|\\section\{|\\maketitle|\\bibliography\{[^}]*\}|\\bibliographystyle\{[^}]*\}|\n\s*\n")
    while i < len(body):
        m = pat.search(body, i)
        if not m:
            blocks.append(("text", body[i:]))
            break
        if m.start() > i:
            blocks.append(("text", body[i:m.start()]))
        tok = m.group(0)
        if tok.startswith("\\begin"):
            env = m.group(1)
            end = body.index("\\end{" + env + "}", m.end())
            blocks.append((env, body[m.end():end], m.group(2) or ""))
            i = end + len("\\end{" + env + "}")
        elif tok.startswith("\\section"):
            close = matching_brace(body, m.end() - 1)
            blocks.append(("section", body[m.end():close]))
            i = close + 1
        elif tok == "\\maketitle":
            blocks.append(("maketitle",))
            i = m.end()
        elif tok.startswith("\\bibliography{"):
            blocks.append(("bibliography",))
            i = m.end()
        else:
            i = m.end()
    return [b for b in blocks if not (b[0] == "text" and not b[1].strip())]


def items_of(content):
    parts = re.split(r"\\item", content)[1:]
    out = []
    for p in parts:
        label = None
        if p.startswith("["):
            close = p.index("]")
            label, p = p[1:close], p[close + 1:]
        out.append((label, p))
    return out


def list_flowable(kind, content, opts, ctx):
    """itemize (bullets), enumerate (numbers, or RQ1.. when the label says RQ) and description (bold run-in labels)."""
    bold = SERIF + "-Bold" if SERIF == "Serif" else "Times-Bold"
    out = []
    for n, (label, text) in enumerate(items_of(content), 1):
        if kind == "description":
            st = ParagraphStyle("dl", parent=S["item"], leftIndent=14, firstLineIndent=-14)
            out.append(Paragraph(f"<b>{inline(label or '', ctx)}</b>&nbsp; {inline(text, ctx)}", st))
        elif kind == "enumerate":
            st = ParagraphStyle("ol", parent=S["item"], leftIndent=34, bulletIndent=4, bulletFontName=bold)
            out.append(Paragraph(inline(text, ctx), st, bulletText=f"RQ{n}" if "RQ" in opts else f"{n}."))
        else:
            st = ParagraphStyle("ul", parent=S["item"], leftIndent=20, bulletIndent=8, bulletFontName=SERIF)
            out.append(Paragraph(inline(text, ctx), st, bulletText="\u2022"))
    return out + [Spacer(1, 4)]


def column_widths(rows, ctx, total=6.5 * inch, pad=14):
    """Column widths from content, as a browser lays out a table: every column gets at least its longest
    single word (so no word is ever split), and the remaining width goes to the columns that want more,
    in proportion to how much more their unwrapped content needs."""
    bold = SERIF + "-Bold" if SERIF == "Serif" else "Times-Bold"
    plain = lambda c: re.sub(r"<[^>]+>", "", inline(c.strip(), ctx)).replace("&amp;", "&")  # noqa: E731
    cells = [[plain(c) for c in r.split("&")] for r in rows]
    width = lambda text, font: pdfmetrics.stringWidth(text, font, 8.2)  # noqa: E731
    low, high = [], []
    for j in range(len(cells[0])):
        column = [(cells[0][j], bold)] + [(r[j], SERIF) for r in cells[1:] if j < len(r)]
        low.append(max(width(w, f) for text, f in column for w in text.split() or [""]) + pad)
        high.append(max(width(text, f) if f == SERIF else max(width(w, f) for w in text.split() or [""])
                        for text, f in column) + pad)
    high = [max(h, l) for h, l in zip(high, low)]
    spare = total - sum(low)
    if spare <= 0:
        return [w * total / sum(low) for w in low]
    want = [h - l for h, l in zip(high, low)]
    if sum(want) <= spare:  # everything fits unwrapped; give the rest to the label column
        widths = high[:]
        widths[0] += total - sum(high)
        return widths
    return [l + spare * w / sum(want) for l, w in zip(low, want)]


def table_flowable(content, ctx):
    caption = inline(arg(content, "caption") or "", ctx)
    tex = (PAPER / arg(content, "input")).read_text(encoding="utf-8")
    body = tex[tex.index("\\toprule") + len("\\toprule"):tex.index("\\bottomrule")]
    rows = [r.strip() for r in body.replace("\\midrule", "").split("\\\\") if r.strip()]
    data = []
    for k, r in enumerate(rows):
        cells = [inline(c.strip(), ctx) for c in r.split("&")]
        data.append([Paragraph(f"<b>{c}</b>" if k == 0 else c, S["cell"] if j == 0 else S["cellr"]) for j, c in enumerate(cells)])
    widths = column_widths(rows, ctx)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    num = ctx["counters"]["table"] = ctx["counters"]["table"] + 1
    return [KeepTogether([Paragraph(f"<b>Table {num}.</b> {caption}", S["caption"]), Spacer(1, 4), t, Spacer(1, 10)])]


def figure_flowable(content, ctx):
    caption = inline(arg(content, "caption") or "", ctx)
    paths = re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]*)\}", content)
    imgs = []
    for opts, path in paths:
        frac = float(re.search(r"([\d.]+)\\linewidth", opts or "1\\linewidth").group(1))
        img = Image(str(PAPER / path))
        w = 6.5 * inch * frac
        img.drawHeight, img.drawWidth = img.imageHeight * w / img.imageWidth, w
        imgs.append(img)
    row = Table([imgs], colWidths=[6.5 * inch / len(imgs)] * len(imgs))
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    num = ctx["counters"]["figure"] = ctx["counters"]["figure"] + 1
    return [KeepTogether([Spacer(1, 4), row, Spacer(1, 4), Paragraph(f"<b>Figure {num}.</b> {caption}", S["caption"]), Spacer(1, 10)])]


def paragraph_flowables(text, ctx):
    text = text.strip()
    if not text:
        return []
    out = []
    m = re.match(r"\\paragraph\{", text)
    if m:
        close = matching_brace(text, m.end() - 1)
        head = inline(text[m.end():close], ctx)
        return [Paragraph(f"<b>{head}</b>&nbsp; {inline(text[close + 1:], ctx)}", S["body"])]
    rendered = inline(text, ctx)
    if rendered:
        out.append(Paragraph(rendered, S["body"]))
    return out


def references(ctx):
    flow = [Paragraph("References", S["h1"])]
    for k in ctx["refs"].keys:
        e = ctx["refs"].bib[k]
        names = [a for a in authors_of(e) if a != "others"]
        fmt = []
        for a in names:
            last, first = (a.split(",", 1) + [""])[:2] if "," in a else (a.split()[-1], " ".join(a.split()[:-1]))
            fmt.append(f"{first.strip()} {last.strip()}".strip())
        who = ", ".join(fmt[:-1]) + (", and " if len(fmt) > 2 else " and ") + fmt[-1] if len(fmt) > 1 else fmt[0]
        if "others" in e.get("author", ""):
            who = ", ".join(fmt) + ", et al"
        venue = e.get("booktitle") or e.get("journal") or e.get("howpublished") or e.get("publisher") or ""
        title = re.sub(r"[{}]", "", e.get("title", ""))
        extra = inline(e.get("note", ""), ctx) if e.get("note") else ""
        title_markup = f"<i>{title}</i>" if e["type"] == "book" else title
        line = f"[{ctx['refs'].number[k]}] {inline(who, ctx)}. {title_markup}. " + (f"<i>{inline(venue, ctx)}</i>, " if venue else "") + f"{e.get('year', '')}." + (f" {extra}" if extra else "")
        flow.append(Paragraph(line, S["ref"]))
    return flow


def build():
    tex = (PAPER / "numgate.tex").read_text(encoding="utf-8")
    macros = load_macros()
    from datetime import date
    ctx = {"date": date.today().strftime("%B %-d, %Y"), "counters": {"table": 0, "figure": 0}}
    pre, body = tex.split("\\begin{document}")
    body = body.split("\\end{document}")[0]
    body = expand_macros(re.sub(r"(?<!\\)%.*", "", body), macros)
    ctx["labels"] = number_labels(body)
    ctx["refs"] = Refs(parse_bib(), body)

    story = []
    section = 0
    for block in split_blocks(body):
        kind = block[0]
        if kind == "maketitle":
            story += [Spacer(1, 6), Paragraph(inline(expand_macros(arg(pre, "title"), macros), ctx), S["title"]), Spacer(1, 10)]
            story += [Paragraph(inline(arg(pre, "author"), ctx), S["author"]), Spacer(1, 4)]
            story += [Paragraph(inline(arg(pre, "date"), ctx), S["date"]), Spacer(1, 14)]
        elif kind == "abstract":
            story += [Paragraph("Abstract", S["abs_head"]), Spacer(1, 4), Paragraph(inline(block[1], ctx), S["abstract"]), Spacer(1, 10)]
        elif kind == "section":
            section += 1
            story.append(Paragraph(f"{section}&nbsp;&nbsp;{inline(block[1], ctx)}", S["h1"]))
        elif kind in {"itemize", "enumerate", "description"}:
            story += list_flowable(kind, block[1], block[2], ctx)
        elif kind == "table":
            story += table_flowable(block[1], ctx)
        elif kind == "figure":
            story += figure_flowable(block[1], ctx)
        elif kind == "bibliography":
            story += references(ctx)
        elif kind == "text":
            story += paragraph_flowables(block[1], ctx)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(SERIF, 9)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    out = PAPER / "numgate.pdf"
    title = re.sub(r"\\\\|[{}]", " ", arg(pre, "title"))
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                            title=re.sub(r"\s+", " ", title).strip(), author=re.split(r"\\\\", arg(pre, "author"))[0].strip())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == "__main__":
    print("Wrote", build())
