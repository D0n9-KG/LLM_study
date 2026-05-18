from dataclasses import dataclass
from pathlib import Path
import re
from config import WORKDIR

SKILLS_DIR = WORKDIR / "skills"

@dataclass
class SkillMainfest:
    name: str
    description: str
    path: Path

@dataclass
class SkillDocument:
    mainfest: SkillMainfest
    body: str

class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    # 扫描目录，批量加载
    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            mainfest = SkillMainfest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(mainfest=mainfest, body=body.strip())
    
    # 解析skill内容：meta和正文
    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        
        meta = {}
        for line in match.group(1).strip().splitlines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)
    
    # 生成技能摘要列表
    def describe_available(self) -> str:
        if not self.documents:
            return "(no skills available)"
        lines = []
        for name in sorted(self.documents):
            mainfest = self.documents[name].mainfest
            lines.append(f"- {mainfest.name}: {mainfest.description}")
        return "\n".join(lines)
    
    def load_full_text(self, name: str) -> str:
        document = self.documents.get(name)
        if not document:
            known = ",".join(sorted(self.documents)) or "(none)"
            return f"Error: Unkonwn skill '{name}'. Available skills: {known}"

        return (
            f"<skill name=\"{document.mainfest.name}\">\n"
            f"{document.body}\n"
            "</skill>"
        ) 

