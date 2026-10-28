"""폴더끼리 부르는 방향 검사. 아래층이 위층을 부르면 실패한다.

  cad  →  motion  →  vision  →  yolo  →  pendant  →  launcher
  (아래)                                              (위)

  cad      물체 CAD 모델 (yaml · STL)
  motion   팔: IK, 제어 루프, 학생용 Arm, 카메라에 등록 요청하는 약속(register_link)
  vision   카메라: 영상 · 뎁스 · 캘리브. 손눈 캘리브는 팔 자세가 필요해서 motion 을 부른다.
  yolo     물체 인식 · 위치 맞추기 (카메라 창 roi_cloud)
  pendant  펜던트 화면
  launcher 버튼 런처 · 반복 데모 (펜던트 글꼴·테마를 같이 쓴다)

위층은 아래층을 자유롭게 부른다. 함수 안에서 늦게 부르는 import 도 같이 본다.
"""

import ast
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LAYERS = ["cad", "motion", "vision", "yolo", "pendant", "launcher"]
# pendant 는 패키지가 아니라 sys.path 에 폴더를 넣고 부른다 (from teach_logic import ...).
PENDANT_MODULES = {p.stem for p in (PROJECT / "pendant").glob("*.py")}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def _layer_of(module: str) -> str | None:
    top = module.split(".")[0]
    if top in LAYERS:
        return top
    if top in PENDANT_MODULES:
        return "pendant"
    return None


class LayerTest(unittest.TestCase):
    def test_no_upward_imports(self):
        bad = []
        for own in LAYERS:
            for path in sorted((PROJECT / own).rglob("*.py")):
                for lineno, module in _imports(path):
                    target = _layer_of(module)
                    if target is not None and LAYERS.index(target) > LAYERS.index(own):
                        bad.append(f"{path.relative_to(PROJECT)}:{lineno}  {own} → {module}")
        self.assertEqual(bad, [], "아래층이 위층을 부릅니다:\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
