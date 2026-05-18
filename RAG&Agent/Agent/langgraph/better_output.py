from dataclasses import asdict, is_dataclass
from IPython.display import display, HTML
import json
import html

def to_jsonable(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "__dict__"):
        return {k: to_jsonable(v) for k, v in x.__dict__.items()}
    return x

def show_whole(obj, title="history"):
    pretty = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2)
    display(HTML(f"""
    <div style="
        border:1px solid #444;
        border-radius:8px;
        padding:12px;
        margin:8px 0;
        background:#1e1e1e;
        color:#d4d4d4;
    ">
      <div style="font-weight:600; margin-bottom:8px;">{html.escape(title)}</div>
      <pre style="
          white-space:pre-wrap;
          word-break:break-word;
          line-height:1.45;
          margin:0;
          font-family:Consolas, 'Courier New', monospace;
          font-size:13px;
      ">{html.escape(pretty)}</pre>
    </div>
    """))