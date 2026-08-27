# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
用于构建 Ultralytics 文档参考部分的辅助文件。.

此脚本递归遍历 ultralytics 目录，根据类和函数构建 *.md 文件组成的 MkDocs 参考部分，
同时创建供 mkdocs.yaml 使用的导航菜单。

注意：必须从仓库根目录运行，不要从 docs 目录运行。
"""

from __future__ import annotations

import ast
import html
import re
import subprocess
import textwrap
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultralytics.utils import LOGGER
from ultralytics.utils.tqdm import TQDM

# 常量
FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[1]
PACKAGE_DIR = REPO_ROOT / "ultralytics"
REFERENCE_DIR = PACKAGE_DIR.parent / "docs/en/reference"
GITHUB_REPO = "ultralytics/ultralytics"
SIGNATURE_LINE_LENGTH = 120

MKDOCS_YAML = PACKAGE_DIR.parent / "mkdocs.yml"
INCLUDE_SPECIAL_METHODS = {
    "__call__",
    "__dir__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__next__",
    "__getattr__",
}
PROPERTY_DECORATORS = {"property", "cached_property"}
CLASS_DEF_RE = re.compile(r"(?:^|\n)class\s(\w+)(?:\(|:)")
FUNC_DEF_RE = re.compile(r"(?:^|\n)(?:async\s+)?def\s(\w+)\(")
SECTION_ENTRY_RE = re.compile(r"([\w*]+)\s*(?:\(([^)]+)\))?:\s*(.*)")
RETURNS_RE = re.compile(r"([^:]+):\s*(.*)")


@dataclass
class ParameterDoc:
    """参数、属性和异常的结构化文档。."""

    name: str
    type: str | None
    description: str
    default: str | None = None


@dataclass
class ReturnDoc:
    """返回值和 yield 值的结构化文档。."""

    type: str | None
    description: str


@dataclass
class ParsedDocstring:
    """Google 风格文档字符串的规范化表示。."""

    summary: str = ""
    description: str = ""
    params: list[ParameterDoc] = field(default_factory=list)
    attributes: list[ParameterDoc] = field(default_factory=list)
    returns: list[ReturnDoc] = field(default_factory=list)
    yields: list[ReturnDoc] = field(default_factory=list)
    raises: list[ParameterDoc] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class DocItem:
    """表示一个有文档说明的符号（类、函数、方法或属性）。."""

    name: str
    qualname: str
    kind: Literal["class", "function", "method", "property"]
    signature: str
    doc: ParsedDocstring
    signature_params: list[ParameterDoc]
    lineno: int
    end_lineno: int
    bases: list[str] = field(default_factory=list)
    children: list[DocItem] = field(default_factory=list)
    module_path: str = ""
    source: str = ""


@dataclass
class DocumentedModule:
    """Python 模块中所有文档项的容器。."""

    path: Path
    module_path: str
    classes: list[DocItem]
    functions: list[DocItem]


# --------------------------------------------------------------------------------------------- #
# 参考文档存根的占位符（旧版）生成
# --------------------------------------------------------------------------------------------- #


def extract_classes_and_functions(filepath: Path) -> tuple[list[str], list[str]]:
    """从 Python 文件中提取顶层类和同步/异步函数名称。."""
    content = filepath.read_text()
    classes = CLASS_DEF_RE.findall(content)
    functions = FUNC_DEF_RE.findall(content)
    return classes, functions


def _with_reference_title(header_content: str, module_path: str) -> str:
    """向参考文档前置元数据中注入简洁且靠前的 `title:`（幂等操作）。.

    H1 保留完整模块路径；`<title>` 使用 `{module} API Reference`（去掉多余的软件包前缀）， 这样文档渲染器追加 ` | Ultralytics` 品牌后缀后仍能满足 60 字符的 SEO
    目标；最深的少数模块路径 仍需依赖渲染器的截断保护。精选的 description/keywords 保持不变。
    """
    if re.search(r"(?m)^title\s*:", header_content):  # line-anchored: ignore `title:` inside a description
        return header_content
    title = f"{module_path.removeprefix(f'{PACKAGE_DIR.name}.')} API Reference"
    return header_content.replace("---\n", f"---\ntitle: {title}\n", 1)


def _existing_frontmatter(md_filepath: Path) -> str:
    """返回页面开头的 YAML 前置元数据块；没有时返回空字符串。.

    匹配位置固定在文件开头：按每个 `---` 分割也会匹配 Markdown 表格分隔线， 当生成器处理自身输出而不是全新克隆的存根时，会把页面内容错误地并入头部。
    """
    if not md_filepath.exists():
        return ""
    match = re.match(r"---\n.*?\n---\n", md_filepath.read_text(), flags=re.DOTALL)
    return f"{match.group()}\n" if match else ""


def create_placeholder_markdown(py_filepath: Path, module_path: str, classes: list[str], functions: list[str]) -> Path:
    """创建最小 Markdown 参考存根。."""
    md_filepath = REFERENCE_DIR / py_filepath.relative_to(PACKAGE_DIR).with_suffix(".md")

    header_content = _existing_frontmatter(md_filepath)
    if not header_content:
        header_content = (
            f"---\ndescription: Reference for `{module_path}` in the Ultralytics package.\n"
            f"keywords: Ultralytics, {module_path}, API reference, YOLO, Python\n---\n\n"
        )
    header_content = _with_reference_title(header_content, module_path)

    module_path_dots = module_path
    module_path_fs = module_path.replace(".", "/")
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path_fs}.py"
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")

    title_content = f"# Reference for `{module_path_fs}.py`\n\n" + contribution_admonition(
        pretty, url, kind="success", title="Improvements"
    )

    md_content = ["<br>\n\n"]
    md_content.extend(f"## ::: {module_path_dots}.{cls}\n\n<br><br><hr><br>\n\n" for cls in classes)
    md_content.extend(f"## ::: {module_path_dots}.{func}\n\n<br><br><hr><br>\n\n" for func in functions)
    if md_content[-1:]:
        md_content[-1] = md_content[-1].replace("<hr><br>\n\n", "")

    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(header_content + title_content + "".join(md_content) + "\n")

    return _relative_to_workspace(md_filepath)


def _get_source(src: str, node: ast.AST) -> str:
    """返回 AST 节点的源代码片段，并提供安全的回退方案。."""
    segment = ast.get_source_segment(src, node)
    if segment:
        return segment
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _format_annotation(annotation: ast.AST | None, src: str) -> str | None:
    """将类型注解格式化为紧凑字符串。."""
    if annotation is None:
        return None
    text = _get_source(src, annotation).strip()
    return " ".join(text.split()) if text else None


def _format_default(default: ast.AST | None, src: str) -> str | None:
    """格式化默认值表达式以供显示。."""
    if default is None:
        return None
    text = _get_source(src, default).strip()
    return " ".join(text.split()) if text else None


def _format_parameter(arg: ast.arg, default: ast.AST | None, src: str) -> str:
    """渲染带有注解和默认值的单个参数。."""
    annotation = _format_annotation(arg.annotation, src)
    rendered = arg.arg
    if annotation:
        rendered += f": {annotation}"
    default_value = _format_default(default, src)
    if default_value is not None:
        rendered += f" = {default_value}" if annotation else f"={default_value}"  # PEP 8 spacing
    return rendered


def collect_signature_parameters(args: ast.arguments, src: str, *, skip_self: bool = True) -> list[ParameterDoc]:
    """从 ast.arguments 对象中收集带类型和默认值的参数。."""
    params: list[ParameterDoc] = []

    def add_param(arg: ast.arg, default_value: ast.AST | None = None):
        """追加参数项，可选择跳过 self/cls。."""
        name = arg.arg
        if skip_self and name in {"self", "cls"}:
            return
        params.append(
            ParameterDoc(
                name=name,
                type=_format_annotation(arg.annotation, src),
                description="",
                default=_format_default(default_value, src),
            )
        )

    posonly = list(getattr(args, "posonlyargs", []))
    regular = list(getattr(args, "args", []))
    defaults = list(getattr(args, "defaults", []))
    total_regular = len(posonly) + len(regular)
    default_offset = total_regular - len(defaults)

    combined = posonly + regular
    for idx, arg in enumerate(combined):
        default = defaults[idx - default_offset] if idx >= default_offset else None
        add_param(arg, default)

    vararg = getattr(args, "vararg", None)
    if vararg:
        add_param(vararg)
        params[-1].name = f"*{params[-1].name}"

    kwonly = list(getattr(args, "kwonlyargs", []))
    kw_defaults = list(getattr(args, "kw_defaults", []))
    for kwarg, default in zip(kwonly, kw_defaults):
        add_param(kwarg, default)

    kwarg = getattr(args, "kwarg", None)
    if kwarg:
        add_param(kwarg)
        params[-1].name = f"**{params[-1].name}"

    return params


def format_signature(
    node: ast.AST, src: str, *, is_class: bool = False, is_async: bool = False, display_name: str | None = None
) -> str:
    """为类、函数和方法构建可读的签名字符串。."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""

    if isinstance(node, ast.ClassDef):
        # 只有模块内继承链的任何位置都不存在 __init__ 时，parse_class 才会传入 ClassDef，因此使用 `Name()`。
        args = ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
        )
    else:
        args = node.args
    name = display_name or getattr(node, "name", "")
    params: list[str] = []

    posonly = list(getattr(args, "posonlyargs", []))
    regular = list(getattr(args, "args", []))
    defaults = list(getattr(args, "defaults", []))
    total_regular = len(posonly) + len(regular)
    default_offset = total_regular - len(defaults)

    combined = posonly + regular
    pairs = [
        (arg, defaults[idx - default_offset] if idx >= default_offset else None) for idx, arg in enumerate(combined)
    ]
    # 构造函数以 Class(...) 的形式调用，因此删除 __init__ 开头绑定的参数，而且只能删除这一个，
    # 因为其他位置的 `cls` 是真实参数（例如 BOTrack(xywh, score, cls, feat=None)）。
    if is_class and pairs and pairs[0][0].arg in {"self", "cls"}:
        pairs, posonly = pairs[1:], posonly[1:]
    for idx, (arg, default) in enumerate(pairs):
        params.append(_format_parameter(arg, default, src))
        if posonly and idx == len(posonly) - 1:
            params.append("/")

    vararg = getattr(args, "vararg", None)
    if vararg:
        rendered = _format_parameter(vararg, None, src)
        params.append(f"*{rendered}")

    kwonly = list(getattr(args, "kwonlyargs", []))
    kw_defaults = list(getattr(args, "kw_defaults", []))
    if kwonly:
        if not vararg:
            params.append("*")
        for kwarg, default in zip(kwonly, kw_defaults):
            params.append(_format_parameter(kwarg, default, src))

    kwarg = getattr(args, "kwarg", None)
    if kwarg:
        rendered = _format_parameter(kwarg, None, src)
        params.append(f"**{rendered}")

    return_annotation = (
        _format_annotation(node.returns, src)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns and not is_class
        else None
    )

    prefix = "" if is_class else ("async def " if is_async else "def ")
    signature = f"{prefix}{name}({', '.join(params)})"
    if return_annotation:
        signature += f" -> {return_annotation}"

    if len(signature) <= SIGNATURE_LINE_LENGTH or not params:
        return signature
    if is_class:  # 长构造函数的原始源码是 `def __init__(self, ...)`，而不是调用形式
        return "{}(\n    {},\n)".format(name, ",\n    ".join(params))

    raw_signature = _get_definition_signature(node, src)
    return raw_signature or signature


def _split_section_entries(lines: list[str]) -> list[list[str]]:
    """根据缩进将文档字符串章节拆分为多个条目。."""
    entries: list[list[str]] = []
    current: list[str] = []
    base_indent: int | None = None

    for raw_line in lines:
        if not raw_line.strip():
            if current:
                current.append("")
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if base_indent is None:
            base_indent = indent
        if indent <= base_indent and current:
            entries.append(current)
            current = [raw_line]
        else:
            current.append(raw_line)
    if current:
        entries.append(current)
    return entries


def _parse_named_entries(lines: list[str]) -> list[ParameterDoc]:
    """解析 Args/Attributes/Raises 风格的章节。."""
    entries = []
    for block in _split_section_entries(lines):
        text = textwrap.dedent("\n".join(block)).strip()
        if not text:
            continue
        first_line, *rest = text.splitlines()
        match = SECTION_ENTRY_RE.match(first_line)
        if match:
            name, type_hint, desc = match.groups()
            # 减少续行缩进，使 _normalize_text 能重新排列换行句子；列表和代码保留原有换行。
            description = "\n".join([" ".join(desc.split()), textwrap.dedent("\n".join(rest))]).strip()
            entries.append(ParameterDoc(name=name, type=type_hint, description=_normalize_text(description)))
        else:
            entries.append(ParameterDoc(name=text, type=None, description=""))
    return entries


def _parse_returns(lines: list[str]) -> list[ReturnDoc]:
    """解析 Returns/Yields 章节。."""
    entries = []
    for block in _split_section_entries(lines):
        text = textwrap.dedent("\n".join(block)).strip()
        if not text:
            continue
        first_line, *rest = text.splitlines()
        match = RETURNS_RE.match(first_line)
        if match:
            type_hint, desc = match.groups()
            cleaned_type = type_hint.strip()
            if cleaned_type.startswith("(") and cleaned_type.endswith(")"):
                cleaned_type = cleaned_type[1:-1].strip()
            # 续行包含句子的其余部分；减少其缩进，使 _normalize_text 能重新排列段落。
            description = "\n".join([desc.strip(), textwrap.dedent("\n".join(rest))]).strip()
            entries.append(ReturnDoc(type=cleaned_type, description=_normalize_text(description)))
        else:
            entries.append(ReturnDoc(type=None, description=_normalize_text(text)))
    return entries


SECTION_ALIASES = {
    "args": "params",
    "arguments": "params",
    "parameters": "params",
    "params": "params",
    "returns": "returns",
    "return": "returns",
    "yields": "yields",
    "yield": "yields",
    "raises": "raises",
    "exceptions": "raises",
    "exception": "raises",
    "attributes": "attributes",
    "attr": "attributes",
    "examples": "examples",
    "example": "examples",
    "notes": "notes",
    "note": "notes",
    "references": "references",
    "reference": "references",
    "methods": "methods",
}


def _normalize_text(text: str) -> str:
    """规范化文本，同时保留表格、提示框和代码块等 Markdown 结构。."""
    if not text:
        return ""
    # 检查文本是否包含需要保留换行的 Markdown 结构。表格检查限定在行首：
    # 纯粹的竖线通常是正文中的联合类型（"Array[M, 4] | Array[M, 5]"），而不是表格。
    if re.search(r"(?m)^\s*\|", text) or any(
        marker in text for marker in ("!!!", "```", "\n#", "\n- ", "\n* ", "\n1. ", "\n    ")
    ):
        # 保留 Markdown 格式，仅删除各行末尾的空白
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()
    # 简单文本：合并段落中的单个换行
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_google_docstring(docstring: str | None) -> ParsedDocstring:
    """将 Google 风格文档字符串解析为结构化数据。."""
    if not docstring:
        return ParsedDocstring()

    lines = textwrap.dedent(docstring).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ParsedDocstring()

    summary = _normalize_text(lines[0].strip())
    body = lines[1:]

    sections: defaultdict[str, list[str]] = defaultdict(list)
    current = "description"
    for line in body:
        stripped = line.strip()
        key = SECTION_ALIASES.get(stripped.rstrip(":").lower())
        if key and stripped.endswith(":"):
            current = key
            continue
        if current != "methods":  # 忽略 "Methods:" 章节；方法会从 AST 渲染
            sections[current].append(line)

    description = "\n".join(sections.pop("description", [])).strip("\n")
    description = _normalize_text(description)

    return ParsedDocstring(
        summary=summary,
        description=description,
        params=_parse_named_entries(sections.get("params", [])),
        attributes=_parse_named_entries(sections.get("attributes", [])),
        returns=_parse_returns(sections.get("returns", [])),
        yields=_parse_returns(sections.get("yields", [])),
        raises=_parse_named_entries(sections.get("raises", [])),
        notes=[textwrap.dedent("\n".join(sections.get("notes", []))).strip()] if sections.get("notes") else [],
        examples=[textwrap.dedent("\n".join(sections.get("examples", []))).strip()] if sections.get("examples") else [],
        references=[line.strip() for line in sections.get("references", []) if line.strip()],
    )


def merge_docstrings(base: ParsedDocstring, extra: ParsedDocstring, ignore_summary: bool = True) -> ParsedDocstring:
    """将 init 文档字符串内容合并到类文档字符串中。."""

    # 保留现有类文档；只有在 init 文档引入新条目时才追加（类文档优先）。
    def _merge_unique(base_items, extra_items, key):
        seen = {key(item) for item in base_items}
        base_items.extend(item for item in extra_items if key(item) not in seen)
        return base_items

    if not base.summary and extra.summary and not ignore_summary:
        base.summary = extra.summary
    if extra.description:
        base.description = "\n\n".join(filter(None, [base.description, extra.description]))
    _merge_unique(base.params, extra.params, lambda p: (p.name, p.type, p.description, p.default))
    _merge_unique(base.attributes, extra.attributes, lambda p: (p.name, p.type, p.description, p.default))
    _merge_unique(base.returns, extra.returns, lambda r: (r.type, r.description))
    _merge_unique(base.yields, extra.yields, lambda r: (r.type, r.description))
    _merge_unique(base.raises, extra.raises, lambda r: (r.name, r.type, r.description, r.default))
    _merge_unique(base.notes, extra.notes, lambda n: n.strip())
    _merge_unique(base.examples, extra.examples, lambda e: e.strip())
    _merge_unique(base.references, extra.references, lambda r: r.strip())
    return base


def _should_document(name: str, *, allow_private: bool = False) -> bool:
    """根据符号名称决定是否包含该符号。."""
    if name in INCLUDE_SPECIAL_METHODS:
        return True
    if name.startswith("_"):
        return allow_private
    return True


def _collect_source_block(src: str, node: ast.AST, end_line: int | None = None) -> str:
    """返回给定节点的去缩进源代码片段，可选择指定结束行。."""
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = src.splitlines()
    # 如果存在装饰器，则从第一个装饰器行开始包含
    decorator_lines = [getattr(d, "lineno", node.lineno) for d in getattr(node, "decorator_list", [])]
    start_line = min([*decorator_lines, node.lineno]) if decorator_lines else node.lineno
    start = max(start_line - 1, 0)
    end = end_line or getattr(node, "end_lineno", node.lineno)
    snippet = "\n".join(lines[start:end])
    return textwrap.dedent(snippet).rstrip()


def _get_definition_signature(node: ast.AST, src: str) -> str:
    """如果源代码中存在，则返回原始的多行定义签名。."""
    if not hasattr(node, "lineno"):
        return ""
    lines = src.splitlines()[node.lineno - 1 :]
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        collected.append(line)
        if stripped.endswith(":"):
            break
    header = textwrap.dedent("\n".join(collected)).rstrip()
    return header[:-1].rstrip() if header.endswith(":") else header


def parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_path: str,
    src: str,
    *,
    parent: str | None = None,
    allow_private: bool = False,
) -> DocItem | None:
    """将函数或方法节点解析为 DocItem。."""
    raw_docstring = ast.get_docstring(node)
    if not _should_document(node.name, allow_private=allow_private) and not raw_docstring:
        return None

    is_async = isinstance(node, ast.AsyncFunctionDef)
    doc = parse_google_docstring(raw_docstring)
    qualname = f"{module_path}.{node.name}" if not parent else f"{parent}.{node.name}"
    decorators = {_get_source(src, d).split(".")[-1] for d in node.decorator_list}
    kind: Literal["function", "method", "property"] = "method" if parent else "function"
    if decorators & PROPERTY_DECORATORS:
        kind = "property"

    signature_params = collect_signature_parameters(node.args, src, skip_self=bool(parent))

    return DocItem(
        name=node.name,
        qualname=qualname,
        kind=kind,
        signature=format_signature(node, src, is_async=is_async),
        doc=doc,
        signature_params=signature_params,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        bases=[],
        children=[],
        module_path=module_path,
        source=_collect_source_block(src, node),
    )


def _class_init(node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """如果类声明了自己的 __init__，则返回它。."""
    return next(
        (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"), None
    )


def _mro(node: ast.ClassDef, class_nodes: dict[str, ast.ClassDef], stack: tuple[str, ...] = ()) -> list[ast.ClassDef]:
    """返回类相对于同一模块中定义的基类的 C3 线性化结果。.

    这里无法解析其他模块中的基类，因此会将其移除；在少数多重继承结构中，这会重新排列剩余的模块内类， 因为缺失的基类不再阻止兄弟类成为下一个候选头。解析这些基类需要导入信息，而不是 AST。
    遇到循环或不一致的继承层次时会提前停止，而不是陷入循环。
    """
    bases = [
        class_nodes[n] for n in (getattr(b, "id", None) for b in node.bases) if n in class_nodes and n not in stack
    ]
    sequences = [_mro(base, class_nodes, (*stack, node.name)) for base in bases] + [bases]
    linearized = [node]
    while any(sequences):
        head = next(
            (s[0] for s in sequences if s and not any(s[0] in rest[1:] for rest in sequences)),
            None,
        )
        if head is None:
            break
        linearized.append(head)
        sequences = [[c for c in s if c is not head] for s in sequences]
    return linearized


def _inherited_init(
    node: ast.ClassDef, class_nodes: dict[str, ast.ClassDef]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """对于未声明 __init__ 的类，按照方法解析顺序返回 Python 绑定的 __init__。."""
    return next((init for base in _mro(node, class_nodes)[1:] if (init := _class_init(base))), None)


def parse_class(node: ast.ClassDef, module_path: str, src: str, class_nodes: dict[str, ast.ClassDef]) -> DocItem:
    """解析类节点，合并 __init__ 文档并收集方法。."""
    class_doc = parse_google_docstring(ast.get_docstring(node))

    own_init = _class_init(node)
    # 未声明 __init__ 的子类仍使用基类签名构造，因此记录基类签名。
    init_node = own_init or _inherited_init(node, class_nodes)
    # 类定义延伸到自身 __init__ 的末尾；如果没有 __init__，则延伸到第一个方法；
    # 对于完全没有方法的类，0 会让 _collect_source_block 使用自身的 end_lineno 回退值。
    first_method = next(
        (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not own_init), None
    )
    class_end = min([first_method.lineno, *(d.lineno for d in first_method.decorator_list)]) - 1 if first_method else 0
    signature_params: list[ParameterDoc] = []
    if init_node:
        init_doc = parse_google_docstring(ast.get_docstring(init_node))
        if init_node is not own_init:
            # 继承的 __init__ 用于记录我们渲染的签名，但其中的说明描述的是基类。
            init_doc = ParsedDocstring(params=init_doc.params)
        class_doc = merge_docstrings(class_doc, init_doc, ignore_summary=True)
        signature_params = collect_signature_parameters(init_node.args, src, skip_self=True)

    bases = [_get_source(src, b) for b in node.bases] if node.bases else []
    signature_node = init_node or node
    class_signature = format_signature(signature_node, src, is_class=True, display_name=node.name)

    methods: list[DocItem] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not own_init:
            method_doc = parse_function(child, module_path, src, parent=f"{module_path}.{node.name}")
            if method_doc:
                methods.append(method_doc)

    return DocItem(
        name=node.name,
        qualname=f"{module_path}.{node.name}",
        kind="class",
        signature=class_signature,
        doc=class_doc,
        signature_params=signature_params,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        bases=bases,
        children=methods,
        module_path=module_path,
        source=_collect_source_block(src, node, end_line=own_init.end_lineno if own_init else class_end),
    )


def parse_module(py_filepath: Path) -> DocumentedModule | None:
    """将 Python 模块解析为结构化文档对象。."""
    try:
        src = py_filepath.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    module_path = (
        f"{PACKAGE_DIR.name}.{py_filepath.relative_to(PACKAGE_DIR).with_suffix('').as_posix().replace('/', '.')}"
    )
    classes: list[DocItem] = []
    functions: list[DocItem] = []

    class_nodes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(parse_class(node, module_path, src, class_nodes))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = parse_function(node, module_path, src, parent=None)
            if func:
                functions.append(func)

    return DocumentedModule(path=py_filepath, module_path=module_path, classes=classes, functions=functions)


def _render_section(title: str, entries: Iterable[str], level: int) -> str:
    """使用给定的标题级别渲染章节。."""
    entries = list(entries)
    if not entries:
        return ""
    heading = f"{'#' * level} {title}\n"
    body = "\n".join(entries).rstrip()
    return f"{heading}{body}\n\n"


def _render_table(headers: list[str], rows: list[list[str]], level: int, title: str | None = None) -> str:
    """渲染带可选标题的 Markdown 表格。."""
    if not rows:
        return ""

    def _clean_cell(value: str | None) -> str:
        """规范化 Markdown 输出的表格单元格值，并转义竖线以使联合类型保持在同一列。."""
        if value is None:
            return ""
        return str(value).replace("\n", "<br>").replace("|", r"\|").strip()

    rows = [[_clean_cell(c) for c in row] for row in rows]
    table_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    heading = f"{'#' * level} {title}\n" if title else ""
    return f"{heading}" + "\n".join(table_lines) + "\n\n"


def _code_fence(source: str, lang: str = "python") -> str:
    """返回带围栏的代码块，可选择用于高亮的语言。."""
    return f"```{lang}\n{source}\n```"


def _merge_params(doc_params: list[ParameterDoc], signature_params: list[ParameterDoc]) -> list[ParameterDoc]:
    """合并文档字符串参数和签名参数，以包含默认值和类型。."""
    sig_map = {p.name.lstrip("*"): p for p in signature_params}
    merged: list[ParameterDoc] = []

    seen = set()
    for dp in doc_params:
        sig = sig_map.get(dp.name.lstrip("*"))
        merged.append(
            ParameterDoc(
                name=dp.name,
                type=dp.type or (sig.type if sig else None),
                description=dp.description,
                default=sig.default if sig else None,
            )
        )
        seen.add(dp.name.lstrip("*"))

    for name, sig in sig_map.items():
        if name in seen:
            continue
        merged.append(sig)

    return merged


DEFAULT_SECTION_ORDER = ["args", "returns", "examples", "notes", "references", "attributes", "yields", "raises"]
SUMMARY_BADGE_MAP = {"Classes": "class", "Properties": "property", "Methods": "method", "Functions": "function"}
_missing_type_warnings: list[str] = []


def contribution_admonition(pretty: str, url: str, *, kind: str = "note", title: str | None = None) -> str:
    """返回标准化的贡献行动提示框。."""
    label = f' "{title}"' if title else ""
    body = (
        f"This page is sourced from [{pretty}]({url}). Have an improvement or example to add? "
        f"Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏"
    )
    return f"!!! {kind}{label}\n\n    {body}\n\n"


def _relative_to_workspace(path: Path) -> Path:
    """如果可行，返回相对于工作区根目录的路径。."""
    try:
        return path.relative_to(PACKAGE_DIR.parent)
    except ValueError:
        return path


def render_source_panel(item: DocItem, module_url: str, module_path: str) -> str:
    """渲染带 GitHub 链接的可折叠源代码面板。."""
    if not item.source:
        return ""
    source_url = f"{module_url}#L{item.lineno}-L{item.end_lineno}"
    summary = f"Source code in <code>{html.escape(module_path)}.py</code>"
    return (
        "<details>\n"
        f"<summary>{summary}</summary>\n\n"
        f'<a href="{source_url}">View on GitHub</a>\n'
        f"{_code_fence(item.source)}\n"
        "</details>\n"
    )


def render_docstring(
    doc: ParsedDocstring,
    level: int,
    signature_params: list[ParameterDoc] | None = None,
    section_order: list[str] | None = None,
    extra_sections: dict[str, str] | None = None,
) -> str:
    """将 ParsedDocstring 转换为 Markdown，并生成类似 mkdocstrings 的表格。."""
    parts: list[str] = []
    if doc.summary:
        parts.append(doc.summary)
    if doc.description:
        parts.append(doc.description)

    sig_params = signature_params or []
    merged_params = _merge_params(doc.params, sig_params)

    sections: dict[str, str] = {}

    # 如果表格的 Type 和 Description 单元格全部为空，则它只重复上方的签名，不包含其他内容。
    if merged_params and any(p.type or p.description.strip() for p in merged_params):
        rows = []
        for p in merged_params:
            default_val = f"`{p.default}`" if p.default not in (None, "") else "*required*"
            rows.append(
                [
                    f"`{p.name}`",
                    f"`{p.type}`" if p.type else "",
                    p.description.strip() if p.description else "",
                    default_val,
                ]
            )
        table = _render_table(["Name", "Type", "Description", "Default"], rows, level, title=None)
        sections["args"] = f"**Args**\n\n{table}"

    if doc.returns:
        rows = []
        for r in doc.returns:
            rows.append([f"`{r.type}`" if r.type else "", r.description])
        table = _render_table(["Type", "Description"], rows, level, title=None)
        sections["returns"] = f"**Returns**\n\n{table}"

    if doc.examples:
        # Google 风格的 Examples 会将说明文字与 >>> 代码交错排列；仅为代码段添加围栏，以保留说明文字。
        blocks: list[str] = []
        for paragraph in (p.strip() for example in doc.examples for p in re.split(r"\n\s*\n", example.strip())):
            lines = paragraph.splitlines()
            prompt = next((i for i, line in enumerate(lines) if line.lstrip().startswith(">>>")), 0)
            if prompt:
                blocks.append("\n".join(lines[:prompt]).strip())
            if paragraph:
                blocks.append(_code_fence("\n".join(lines[prompt:]).strip()))
        if blocks:
            sections["examples"] = "**Examples**\n\n" + "\n\n".join(blocks) + "\n\n"

    if doc.notes:
        note_text = "\n\n".join(doc.notes).strip()
        indented = textwrap.indent(note_text, "    ")
        sections["notes"] = f'!!! note "Notes"\n\n{indented}\n\n'

    if doc.attributes:
        rows = []
        for a in doc.attributes:
            rows.append(
                [f"`{a.name}`", f"`{a.type}`" if a.type else "", a.description.strip() if a.description else ""]
            )
        table = _render_table(["Name", "Type", "Description"], rows, level, title=None)
        sections["attributes"] = f"**Attributes**\n\n{table}"

    if doc.yields:
        rows = []
        for r in doc.yields:
            rows.append([f"`{r.type}`" if r.type else "", r.description])
        table = _render_table(["Type", "Description"], rows, level, title=None)
        sections["yields"] = f"**Yields**\n\n{table}"

    if doc.raises:
        rows = []
        for e in doc.raises:
            type_cell = e.type or e.name
            rows.append([f"`{type_cell}`" if type_cell else "", e.description or ""])
        table = _render_table(["Type", "Description"], rows, level, title=None)
        sections["raises"] = f"**Raises**\n\n{table}"

    if doc.references:
        links = "\n".join(ref if ref.startswith("- ") else f"- {ref}" for ref in doc.references)
        sections["references"] = f"**References**\n\n{links}\n\n"

    if extra_sections:
        sections.update({k: v for k, v in extra_sections.items() if v})
    # 确保章节顺序中的条目唯一，避免重复渲染（例如类注入 "examples" 时）。
    order = list(dict.fromkeys(section_order or DEFAULT_SECTION_ORDER))

    ordered_sections: list[str] = []
    seen = set()
    for key in order:
        section = sections.get(key)
        if section:
            ordered_sections.append(section)
            seen.add(key)

    for key, section in sections.items():
        if key not in seen:
            ordered_sections.append(section)

    parts.extend(filter(None, ordered_sections))
    return "\n\n".join([p.rstrip() for p in parts if p]).strip() + ("\n\n" if parts else "")


def item_anchor(item: DocItem) -> str:
    """为文档项创建稳定的锚点。."""
    return item.qualname


def display_qualname(item: DocItem) -> str:
    """返回用于显示的清理后完全限定名称（去除 __init__ 噪声）。."""
    return item.qualname.replace(".__init__.", ".")


def render_summary_tabs(module: DocumentedModule) -> str:
    """渲染类、方法和函数的选项卡摘要，便于快速导航。."""
    tab_entries: list[tuple[str, list[str]]] = []

    if module.classes:
        tab_entries.append(
            (
                "Classes",
                [f"- [`{cls.name}`](#{item_anchor(cls)})" for cls in module.classes],
            )
        )

    property_links = []
    method_links = []
    for cls in module.classes:
        for child in cls.children:
            if child.kind == "property":
                property_links.append(f"- [`{cls.name}.{child.name}`](#{item_anchor(child)})")
        for child in cls.children:
            if child.kind == "method":
                method_links.append(f"- [`{cls.name}.{child.name}`](#{item_anchor(child)})")
    if property_links:
        tab_entries.append(("Properties", property_links))
    if method_links:
        tab_entries.append(("Methods", method_links))

    if module.functions:
        tab_entries.append(
            (
                "Functions",
                [f"- [`{func.name}`](#{item_anchor(func)})" for func in module.functions],
            )
        )

    if not tab_entries:
        return ""

    lines = ['!!! abstract "Summary"\n']
    for label, bullets in tab_entries:
        badge_class = SUMMARY_BADGE_MAP.get(label, label.lower())
        label_badge = f'<span class="doc-kind doc-kind-{badge_class}">{label}</span>'
        lines.append(f'    === "{label_badge}"\n')
        lines.append("\n".join(f"        {line}" for line in bullets))
        lines.append("")  # 每个选项卡块后添加空行
    return "\n".join(lines).rstrip() + "\n\n"


def render_item(item: DocItem, module_url: str, module_path: str, level: int = 2) -> str:
    """将类、函数或方法渲染为 Markdown。."""
    anchor = item_anchor(item)
    title_prefix = item.kind.capitalize()
    anchor_id = anchor.replace("_", r"\_")  # 转义下划线，使 attr_list 将其保留在 ID 中
    heading = f"{'#' * level} {title_prefix} `{display_qualname(item)}` {{#{anchor_id}}}"
    signature_block = f"```python\n{item.signature}\n```\n"

    parts = [heading, signature_block]

    if item.bases:
        bases = ", ".join(f"`{b}`" for b in item.bases)
        parts.append(f"**Bases:** {bases}\n")

    # 检查签名和文档字符串中缺少类型注解的参数
    if item.signature_params and item.doc.params:
        merged = _merge_params(item.doc.params, item.signature_params)
        missing = [p.name for p in merged if not p.type]
        if missing:
            _missing_type_warnings.append(f"{item.qualname}: {', '.join(missing)}")

    if item.kind == "class":
        method_section = None
        if item.children:
            props = [c for c in item.children if c.kind == "property"]
            methods = [c for c in item.children if c.kind == "method"]
            methods.sort(key=lambda m: (not m.name.startswith("__"), m.name))

            rows = []
            for child in props + methods:
                summary = child.doc.summary or (
                    _normalize_text(child.doc.description).split("\n\n")[0] if child.doc.description else ""
                )
                rows.append([f"[`{child.name}`](#{item_anchor(child)})", summary.strip()])
            if rows:
                table = _render_table(["Name", "Description"], rows, level + 1, title=None)
                method_section = f"**Methods**\n\n{table}"

        order = ["args", "attributes", "methods", "examples", *DEFAULT_SECTION_ORDER]
        rendered = render_docstring(
            item.doc,
            level + 1,
            signature_params=item.signature_params,
            section_order=order,
            extra_sections={"methods": method_section} if method_section else None,
        )
        parts.append(rendered)
    else:
        parts.append(render_docstring(item.doc, level + 1, signature_params=item.signature_params))

    if item.kind == "class" and item.source:
        parts.append(render_source_panel(item, module_url, module_path))

    if item.children:
        props = [c for c in item.children if c.kind == "property"]
        methods = [c for c in item.children if c.kind == "method"]
        methods.sort(key=lambda m: (not m.name.startswith("__"), m.name))

        ordered_children = props + methods
        parts.append("<br>\n")
        for idx, child in enumerate(ordered_children):
            parts.append(render_item(child, module_url, module_path, level + 1))
            if idx != len(ordered_children) - 1:
                parts.append("<br>\n")

    if item.source and item.kind != "class":
        parts.append(render_source_panel(item, module_url, module_path))

    return "\n\n".join(p.rstrip() for p in parts if p).rstrip() + "\n\n"


def render_module_markdown(module: DocumentedModule) -> str:
    """渲染完整的模块参考内容。."""
    module_path = module.module_path.replace(".", "/")
    module_url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"
    content: list[str] = ["<br>\n"]

    summary_tabs = render_summary_tabs(module)
    if summary_tabs:
        content.append(summary_tabs)

    sections: list[str] = []
    for idx, cls in enumerate(module.classes):
        sections.append(render_item(cls, module_url, module_path, level=2))
        if idx != len(module.classes) - 1 or module.functions:
            sections.append("<br><br><hr><br>\n")
    for idx, func in enumerate(module.functions):
        sections.append(render_item(func, module_url, module_path, level=2))
        if idx != len(module.functions) - 1:
            sections.append("<br><br><hr><br>\n")

    content.extend(sections)
    return "\n".join(content).rstrip() + "\n\n<br><br>\n"


def create_markdown(module: DocumentedModule) -> Path:
    """创建包含给定 Python 模块 API 参考的 Markdown 文件。."""
    md_filepath = REFERENCE_DIR / module.path.relative_to(PACKAGE_DIR).with_suffix(".md")
    exists = md_filepath.exists()

    header_content = _existing_frontmatter(md_filepath)
    if not header_content:
        header_content = (
            f"---\ndescription: Reference for `{module.module_path}` in the Ultralytics package.\n"
            f"keywords: Ultralytics, {module.module_path}, API reference, YOLO, Python\n---\n\n"
        )
    header_content = _with_reference_title(header_content, module.module_path)

    module_path_fs = module.module_path.replace(".", "/")
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path_fs}.py"
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # Properly display __init__.py filenames

    title_content = f"# Reference for `{module_path_fs}.py`\n\n" + contribution_admonition(
        pretty, url, kind="success", title="Improvements"
    )

    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(header_content + title_content + render_module_markdown(module))

    if not exists:
        subprocess.run(["git", "add", "-f", str(md_filepath)], check=True, cwd=REPO_ROOT)

    return _relative_to_workspace(md_filepath)


def nested_dict():
    """创建并返回嵌套的 defaultdict。."""
    return defaultdict(nested_dict)


def sort_nested_dict(d: dict) -> dict:
    """递归排序嵌套字典。."""
    return {k: sort_nested_dict(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}


def create_nav_menu_yaml(nav_items: list[str]) -> str:
    """创建并返回导航菜单的 YAML 字符串。."""
    nav_tree = nested_dict()

    for item_str in nav_items:
        item = Path(item_str)
        parts = item.parts
        current_level = nav_tree["reference"]
        for part in parts[2:-1]:  # 跳过 docs/reference 和文件名
            current_level = current_level[part]
        current_level[parts[-1].replace(".md", "")] = item

    def _dict_to_yaml(d, level=0):
        """将嵌套字典转换为带缩进的 YAML 格式字符串。."""
        yaml_str = ""
        indent = "  " * level
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                yaml_str += f"{indent}- {k}:\n{_dict_to_yaml(v, level + 1)}"
            else:
                yaml_str += f"{indent}- {k}: {str(v).replace('docs/en/', '')}\n"
        return yaml_str

    reference_yaml = _dict_to_yaml(sort_nested_dict(nav_tree))
    LOGGER.info(f"Scan complete, generated reference section with {len(reference_yaml.splitlines())} lines")
    return reference_yaml


def extract_document_paths(yaml_section: str) -> list[str]:
    """从 YAML 章节中提取文档路径，忽略格式和结构。."""
    paths = []
    # 匹配 `key: path` 条目
    path_matches = re.findall(r":\s*([^\s][^:\n]*?)(?:\n|$)", yaml_section)
    for path in path_matches:
        path = path.strip()
        if path and not path.startswith("-") and not path.endswith(":"):
            paths.append(path)
    # 同时匹配不带键名的 `- path.md` 条目（例如 `- reference/index.md`）
    paths.extend(re.findall(r"^\s*-\s+([^\s:][^:\n]*\.md)\s*$", yaml_section, re.MULTILINE))
    return sorted(paths)


def update_mkdocs_file(reference_yaml: str) -> None:
    """仅在检测到文档路径变化时，使用新的参考章节更新 mkdocs.yaml 文件。."""
    mkdocs_content = MKDOCS_YAML.read_text()

    # 查找顶层 Reference 章节
    ref_pattern = r"(\n  - Reference:[\s\S]*?)(?=\n  - \w|$)"
    ref_match = re.search(ref_pattern, mkdocs_content)

    # 使用正确缩进构建新章节。手写的 `reference/index.md` 概览固定在顶部，
    # 使 Reference 章节拥有入口页（与 Modes、Tasks、Datasets、Help 等章节的约定一致）。
    # 它必须与下方自动生成的同级条目使用相同的 4 空格内部缩进，
    # 这样生成的 YAML 才可解析（否则同级条目会被嵌套为字符串标量的子项）。
    inner_lines = [line for line in reference_yaml.splitlines() if line.strip() != "- reference:"]
    inner_lines.insert(0, "    - reference/index.md")
    new_section_lines = ["\n  - Reference:", *(f"    {line}" for line in inner_lines)]
    new_ref_section = "\n".join(new_section_lines) + "\n"

    if ref_match:
        # 已找到现有的 Reference 章节
        ref_section = ref_match.group(1)
        LOGGER.info(f"Found existing top-level Reference section ({len(ref_section)} chars)")

        # 仅比较文档路径
        existing_paths = extract_document_paths(ref_section)
        new_paths = extract_document_paths(new_ref_section)

        # 检查文档路径是否相同（忽略结构或格式差异）
        if len(existing_paths) == len(new_paths) and set(existing_paths) == set(new_paths):
            LOGGER.info(f"No changes detected in document paths ({len(existing_paths)} items). Skipping update.")
            return

        LOGGER.info(f"Changes detected: {len(new_paths)} document paths vs {len(existing_paths)} existing")

        # 更新内容
        new_content = mkdocs_content.replace(ref_section, new_ref_section)
        MKDOCS_YAML.write_text(new_content)
        try:
            result = subprocess.run(
                ["npx", "prettier", "--write", str(MKDOCS_YAML)],
                capture_output=True,
                text=True,
                cwd=PACKAGE_DIR.parent,
                check=False,
            )
            if result.returncode != 0:
                LOGGER.warning(f"prettier formatting failed: {result.stderr.strip()}")
        except FileNotFoundError:
            LOGGER.warning("prettier not found (install Node.js or run 'npm i -g prettier'), skipping YAML formatting")
        LOGGER.info(f"Updated Reference section in {MKDOCS_YAML}")
    elif help_match := re.search(r"(\n  - Help:)", mkdocs_content):
        # 不存在 Reference 章节，需要添加
        help_section = help_match.group(1)
        # 插入到 Help 章节之前
        new_content = mkdocs_content.replace(help_section, f"{new_ref_section}{help_section}")
        MKDOCS_YAML.write_text(new_content)
        LOGGER.info(f"Added new Reference section before Help in {MKDOCS_YAML}")
    else:
        LOGGER.warning("Could not find a suitable location to add Reference section")


def _finalize_reference(nav_items: list[str], update_nav: bool, created: int, created_label: str) -> list[str]:
    """可选地同步导航并打印创建摘要。."""
    if update_nav:
        update_mkdocs_file(create_nav_menu_yaml(nav_items))
    if created:
        LOGGER.info(f"Created {created} new {created_label}")
    return nav_items


def build_reference(update_nav: bool = True) -> list[str]:
    """为旧版存根流程创建参考占位文件。."""
    return build_reference_placeholders(update_nav=update_nav)


def build_reference_placeholders(update_nav: bool = True) -> list[str]:
    """创建最小参考占位文件，并可选地更新导航。."""
    nav_items: list[str] = []
    created = 0
    orphans = set(REFERENCE_DIR.rglob("*.md"))
    orphans.discard(REFERENCE_DIR / "index.md")  # Preserve hand-written overview page

    for py_filepath in TQDM(list(PACKAGE_DIR.rglob("*.py")), desc="Building reference stubs", unit="file"):
        classes, functions = extract_classes_and_functions(py_filepath)
        if not classes and not functions:
            continue
        module_path = (
            f"{PACKAGE_DIR.name}.{py_filepath.relative_to(PACKAGE_DIR).with_suffix('').as_posix().replace('/', '.')}"
        )
        md_filepath = REFERENCE_DIR / py_filepath.relative_to(PACKAGE_DIR).with_suffix(".md")
        exists = md_filepath.exists()
        orphans.discard(md_filepath)
        md_rel = create_placeholder_markdown(py_filepath, module_path, classes, functions)
        nav_items.append(str(md_rel))
        if not exists:
            created += 1
    for orphan in orphans:
        orphan.unlink()
    if update_nav:
        update_mkdocs_file(create_nav_menu_yaml(nav_items))
    if created:
        LOGGER.info(f"Created {created} new reference stub files")
    return nav_items


def build_reference_docs(update_nav: bool = False) -> list[str]:
    """渲染完整的基于文档字符串的参考内容。."""
    _missing_type_warnings.clear()
    nav_items: list[str] = []
    created = 0

    desc = f"Docstrings {GITHUB_REPO or PACKAGE_DIR.name}"
    for py_filepath in TQDM(list(PACKAGE_DIR.rglob("*.py")), desc=desc, unit="file"):
        md_target = REFERENCE_DIR / py_filepath.relative_to(PACKAGE_DIR).with_suffix(".md")
        exists_before = md_target.exists()
        module = parse_module(py_filepath)
        if not module or (not module.classes and not module.functions):
            continue
        md_rel_filepath = create_markdown(module)
        if not exists_before:
            created += 1
        nav_items.append(str(md_rel_filepath))

    if update_nav:
        update_mkdocs_file(create_nav_menu_yaml(nav_items))
    if created:
        LOGGER.info(f"Created {created} new reference files")
    if _missing_type_warnings:
        LOGGER.warning(f"{len(_missing_type_warnings)} functions/methods have parameters missing type annotations:")
        for warning in _missing_type_warnings:
            LOGGER.warning(f"  - {warning}")
        raise ValueError(
            f"{len(_missing_type_warnings)} parameters missing types in both signature and docstring. "
            f"Add type annotations to the function signature or (type) in the docstring Args section."
        )
    return nav_items


def build_reference_for(
    package_dir: Path, reference_dir: Path, github_repo: str, update_nav: bool = False
) -> list[str]:
    """临时切换软件包上下文，为另一个项目构建参考文档。."""
    global PACKAGE_DIR, REFERENCE_DIR, GITHUB_REPO
    prev = (PACKAGE_DIR, REFERENCE_DIR, GITHUB_REPO)
    try:
        PACKAGE_DIR, REFERENCE_DIR, GITHUB_REPO = package_dir, reference_dir, github_repo
        return build_reference_docs(update_nav=update_nav)
    finally:
        PACKAGE_DIR, REFERENCE_DIR, GITHUB_REPO = prev


def main():
    """CLI 入口。."""
    build_reference(update_nav=True)


if __name__ == "__main__":
    main()
