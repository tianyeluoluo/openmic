import argparse
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

from openmic.autogen_v02 import AutoGenConfig, AutoGenV02Workflow
from openmic.humor_generation import clean_writer_script
from openmic.humor_adaptation import DEFAULT_ADAPTATION_PROBABILITY
from openmic.models import ProjectRequest
from openmic.workflow import OpenMicWorkflow


STYLE_ALIASES = {
    "观察": "observational",
    "自嘲": "self_deprecating",
    "吐槽": "roast",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Chinese stand-up comedy with the OpenMic workflow"
    )
    parser.add_argument("--topic", required=True, help="生活主题")
    parser.add_argument(
        "--style",
        default="observational",
        choices=["observational", "self_deprecating", "roast", "观察", "自嘲", "吐槽"],
    )
    parser.add_argument("--duration", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--audience", default="大学生")
    parser.add_argument(
        "--context", default="",
        help="可选的真实场景、人物关系和已知事实；创作时不得违背",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "autogen"],
        default="autogen",
        help="mock 无需 Key；autogen 使用课程要求的 AutoGen 0.2 GroupChat",
    )
    parser.add_argument(
        "--content-mode", choices=["baseline", "cfunset"], default="cfunset",
        help="真实 AutoGen 模式的内容策略；默认使用 CFunSet",
    )
    parser.add_argument(
        "--cfunset-index", type=Path, default=None,
        help="prepare 命令生成的 CFunSet 索引路径",
    )
    parser.add_argument("--verbose", action="store_true", help="显示五个 Agent 的完整轨迹")
    parser.add_argument("--adaptation-probability", type=float, default=None,
                        help=f"CFunSet原例改编概率0..1，默认{DEFAULT_ADAPTATION_PROBABILITY}（可用环境变量覆盖）；0参考后自主创作，1每次尝试改编")
    parser.add_argument("--adaptation-seed", type=int, default=None, help="复现改编分支抽样")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = ProjectRequest(
        topic=args.topic,
        style=STYLE_ALIASES.get(args.style, args.style),
        duration_minutes=args.duration,
        audience=args.audience,
        context=args.context,
    )
    if args.backend == "autogen":
        config = AutoGenConfig.from_env()
        index = args.cfunset_index or Path(os.getenv("OPENMIC_CFUNSET_INDEX", "artifacts/cfunset/index.jsonl"))
        with redirect_stdout(io.StringIO()):
            result = AutoGenV02Workflow(
                config,
                content_mode=args.content_mode,
                cfunset_index=index,
                adaptation_probability=args.adaptation_probability,
                adaptation_seed=args.adaptation_seed,
            ).run(request)
    else:
        result = OpenMicWorkflow().run(request)

    if args.verbose:
        for message in result.messages:
            print(f"\n[{message.agent}]\n{message.content}")
    script = next((message.content for message in reversed(result.messages)
                   if message.agent == "JokeWriter"), "")
    if script:
        if result.adaptation_source_id:
            print(f"\n[来源] CFunSet原例改编，原例ID={result.adaptation_source_id}")
        print(f"\n[最终脚本]\n{clean_writer_script(script)}")
    print(
        f"\nResult: approved={result.approved}, score={result.score:.1f}, "
        f"revisions={result.revision_count}"
    )
    if result.validation_issues:
        print("本地脚本检查未通过：" + "；".join(result.validation_issues))


if __name__ == "__main__":
    main()
