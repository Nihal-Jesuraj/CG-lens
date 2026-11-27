"""Build the single-column DOCX edition of the CG Lens report.

The LaTeX file is the canonical source.  This script rewrites the
IEEEtran-specific constructs that pandoc cannot resolve (cross-references,
citations, figure/table numbers, author block) and then hands the result to
pandoc.  Styling is applied afterwards with python-docx.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
TEX = HERE / "CG_Lens_Project_Report.tex"
AUX = HERE / "CG_Lens_Project_Report.aux"
TMP_TEX = Path("/private/tmp/cg_docx_body.tex")
OUT_DOCX = HERE / "CG_Lens_Project_Report.docx"

TITLE = (
    "CG Lens: A Comparative Study of MRI Reconstruction and "
    "Patient-Disjoint SVM Classification"
)
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

AUTHOR = r"""
\begin{center}
{\Large\bfseries Nihal J}\\
BL.SC.U4AIE25230\\
Computer Science and Engineering (AI), Amrita Vishwa Vidyapeetham\\
Bengaluru, India
\end{center}

\begin{center}
{\Large\bfseries Shoham Prasad Padhi}\\
BL.SC.U4AIE25243\\
Computer Science and Engineering (AI), Amrita Vishwa Vidyapeetham\\
Bengaluru, India
\end{center}

\begin{center}
Course 22MAT220 Mathematics for Computing III · Date: November 2026
\end{center}
"""


def read_labels() -> dict[str, str]:
    aux = AUX.read_text()
    out = {}
    for key, val in re.findall(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{", aux):
        val = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", val)
        val = re.sub(r"\\[a-zA-Z]+\s*", "", val)
        val = val.replace("{", "").replace("}", "").strip()
        out[key] = val
    return out


def brace_span(text: str, start: int) -> tuple[str, int]:
    """Return the content of the brace group starting at *start* and its end."""
    depth = 1
    i = start
    buf = []
    while i < len(text) and depth:
        ch = text[i]
        escaped = i > 0 and text[i - 1] == "\\"
        if ch == "{" and not escaped:
            depth += 1
        elif ch == "}" and not escaped:
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf), i + 1


def number_captions(body: str, labels: dict[str, str]) -> str:
    insertions = []
    pos = 0
    opener = r"\caption{"
    while True:
        j = body.find(opener, pos)
        if j < 0:
            break
        text, end = brace_span(body, j + len(opener))
        label = re.search(r"\\label\{([^}]+)\}", body[end:end + 400])
        num = labels.get(label.group(1), "") if label else ""
        before = body[:j]
        is_figure = before.rfind(r"\begin{figure") > before.rfind(r"\begin{table")
        prefix = ""
        if num:
            kind = "Fig." if is_figure else "Table"
            prefix = f"{kind} {num}. "
        insertions.append((j + len(opener), prefix))
        pos = end
    for at, prefix in reversed(insertions):
        body = body[:at] + prefix + body[at:]
    return body


def main() -> None:
    tex = TEX.read_text()
    labels = read_labels()
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]

    body = re.sub(r"\\title\{.*?\}\s*", "", body, count=1, flags=re.S)
    body = re.sub(r"\\author\{.*?\n\}\s*", lambda m: AUTHOR, body, count=1, flags=re.S)
    body = body.replace(r"\maketitle", "")

    # Keep the abstract after the author block: pandoc hoists a LaTeX
    # abstract environment to the top of the document, so emit it as an
    # ordinary bold-led paragraph instead.
    body = body.replace(r"\begin{abstract}", r"\noindent\textbf{Abstract}---", 1)
    body = body.replace(r"\end{abstract}", "", 1)
    body = body.replace(r"\begin{IEEEkeywords}", r"\noindent\textbf{Index Terms}---", 1)
    body = body.replace(r"\end{IEEEkeywords}", "", 1)

    bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", body)
    cite_num = {k: i + 1 for i, k in enumerate(bib_keys)}
    body = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda m: ", ".join(
            f"[{cite_num[k.strip()]}]" for k in m.group(1).split(",") if k.strip() in cite_num
        ),
        body,
    )

    body = number_captions(body, labels)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)

    state = {"section": 0, "subsection": 0, "appendix": 0}

    def section(m):
        state["section"] += 1
        state["subsection"] = 0
        return f"\\section{{{ROMAN[state['section'] - 1]}. {m.group(1)}}}"

    def appendix(m):
        state["appendix"] += 1
        return f"\\section{{Appendix {LETTERS[state['appendix'] - 1]}. {m.group(1)}}}"

    def subsection(m):
        state["subsection"] += 1
        title = m.group(2) if m.group(1) is None else m.group(1)
        return f"\\subsection{{{LETTERS[state['subsection'] - 1]}. {title}}}"

    head, sep, tail = body.partition(r"\appendices")

    def walk(m):
        if m.group(1) is not None:
            return section(m)
        return subsection(m)

    def walk_appendix(m):
        if m.group(1) is not None:
            return appendix(m)
        return subsection(m)

    head = re.sub(r"\\section\{([^{}]*)\}|\\subsection\{([^{}]*)\}", walk, head)
    tail = re.sub(r"\\section\{([^{}]*)\}|\\subsection\{([^{}]*)\}", walk_appendix, tail)
    body = head + tail

    before_bib, sep, bib = body.partition(r"\begin{thebibliography}")
    if sep:
        refs = bib.split(r"\end{thebibliography}", 1)[0]
        refs = refs.split("}", 1)[1]
        entries = re.findall(r"\\bibitem\{[^}]+\}(.*?)(?=\\bibitem|$)", refs, flags=re.S)
        lines = ["\\section*{References}\n"]
        for i, entry in enumerate(entries, 1):
            lines.append(f"[{i}] {' '.join(entry.split())}\n")
        after_bib = bib.split(r"\end{thebibliography}", 1)
        rest = after_bib[1] if len(after_bib) > 1 else ""
        body = before_bib + "\n" + "\n".join(lines) + "\n" + rest
    else:
        body = before_bib

    body = re.sub(r"\\begin\{(figure\*?|table\*?)\}\[[^\]]*\]", r"\\begin{\1}", body)
    # pandoc drops the caption of a starred float, so make every float
    # unstarred; the DOCX edition is single-column anyway.
    body = body.replace(r"\begin{figure*}", r"\begin{figure}").replace(
        r"\end{figure*}", r"\end{figure}"
    )
    body = body.replace(r"\begin{table*}", r"\begin{table}").replace(
        r"\end{table*}", r"\end{table}"
    )
    body = body.replace(
        r"\begin{itemize}[leftmargin=1.1em,itemsep=1pt,topsep=2pt]", r"\begin{itemize}"
    )
    body = re.sub(r"\\(centering|footnotesize|scriptsize|small|normalsize)\b", "", body)
    body = body.replace(r"\etal", "et al.")
    body = body.replace(r"\AH", r"A^{\mathrm{H}}")

    def width(m):
        opt = m.group(1)
        opt = re.sub(
            r"([0-9.]+)\\textwidth", lambda n: f"{float(n.group(1)) * 6.3:.2f}in", opt
        )
        opt = re.sub(
            r"([0-9.]+)\\columnwidth", lambda n: f"{float(n.group(1)) * 6.3:.2f}in", opt
        )
        opt = opt.replace(r"\textwidth", "6.3in").replace(r"\columnwidth", "6.3in")
        return f"\\includegraphics[{opt}]"

    body = re.sub(r"\\includegraphics\[([^\]]*)\]", width, body)

    TMP_TEX.write_text(body)
    subprocess.run(
        [
            "pandoc",
            str(TMP_TEX),
            "-f",
            "latex",
            "-t",
            "docx",
            "--metadata",
            f"title={TITLE}",
            "-o",
            str(OUT_DOCX),
        ],
        check=True,
    )
    style_docx()


def _set_font(run, name="Times New Roman", size=None, bold=None, italic=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def _shade(cell, hex_color="D9E2F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tblPr.append(borders)


def style_docx() -> None:
    doc = docx.Document(OUT_DOCX)

    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.top_margin = Inches(0.9)
    sec.bottom_margin = Inches(0.9)
    sec.left_margin = Inches(1.0)
    sec.right_margin = Inches(1.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.12

    for name, size in (("Heading 1", 13), ("Heading 2", 11.5)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(4)

    title_idx = None
    for i, p in enumerate(doc.paragraphs):
        style = p.style.name
        if style == "Title":
            title_idx = i
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                _set_font(r, size=19, bold=True)
        elif style == "Author":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                _set_font(r, size=11.5, bold=True)
        elif style in ("Image Caption", "Table Caption"):
            for r in p.runs:
                _set_font(r, size=9, italic=(style == "Image Caption"))
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
        elif style.startswith("Heading"):
            for r in p.runs:
                _set_font(r, bold=True)
        else:
            for r in p.runs:
                _set_font(r)
                if r.font.size is None:
                    r.font.size = Pt(10.5)

    if title_idx is not None:
        for p in doc.paragraphs[title_idx + 1: title_idx + 4]:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Keep code indentation: some viewers collapse leading ASCII spaces
    # after a soft line break, so use non-breaking spaces at line starts.
    for p in doc.paragraphs:
        if p.style.name != "Source Code":
            continue
        at_line_start = True
        for r in p.runs:
            if r.text == "\n":
                at_line_start = True
                continue
            if at_line_start and r.text.startswith(" "):
                n = len(r.text) - len(r.text.lstrip(" "))
                r.text = "\u00a0" * n + r.text[n:]
            at_line_start = False
            _set_font(r, name="Courier New", size=8)

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        _borders(table)
        for row_i, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        _set_font(r, size=9, bold=(row_i == 0))
                if row_i == 0:
                    _shade(cell, "EDF2F9")

    props = doc.core_properties
    props.title = TITLE
    props.author = "Nihal J; Shoham Prasad Padhi"
    props.subject = "22MAT220 Mathematics for Computing III - Project Report"
    props.keywords = (
        "MRI reconstruction, total variation, conjugate gradient, IRLS, "
        "SVM, SMO, patient-disjoint validation"
    )
    props.comments = "CG Lens project report, November 2026"

    doc.save(OUT_DOCX)
    print("wrote", OUT_DOCX)


if __name__ == "__main__":
    main()
