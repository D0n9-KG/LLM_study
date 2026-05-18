from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic

load_dotenv()

WORKDIR = Path.cwd()
client = Anthropic(
    base_url="http://localhost:4000/anthropic",
    api_key="any-string"
)

TRUST_MARKER = WORKDIR / ".claude" / ".claude_trusted"

MODEL = "deepseek-chat"

MODES = ("default", "plan", "auto")

SKILLS_DIR = WORKDIR / "skills"

TASKS_DIR = WORKDIR / "tasks"

RUNTIME_DIR = WORKDIR / ".runtime-tasks"