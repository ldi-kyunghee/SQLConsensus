# SQLConsensus

Official implementation of "SQLConsensus: Confidence Estimation in Text-to-SQL from Structural and Execution Agreement."

## Install

```bash
uv sync
```

## Prepare data

```bash
uv run sqlconsensus prepare bird --dataset BIRD/questions.json --db-dir BIRD/databases --out data/bird.json
uv run sqlconsensus prepare spider --dataset Spider/questions.json --db-dir Spider/databases --out data/spider.json
uv run sqlconsensus prepare ehrsql --dataset EHRSQL/mimic_iii/questions.json --db-file EHRSQL/mimic_iii/mimic_iii.sqlite --database mimic_iii --omnisql-repo OmniSQL --out data/ehrsql_mimic_iii.json
uv run sqlconsensus prepare ehrsql --dataset EHRSQL/eicu/questions.json --db-file EHRSQL/eicu/eicu.sqlite --database eicu --omnisql-repo OmniSQL --out data/ehrsql_eicu.json
```

## Run evaluation: open-weight model

```bash
uv run sqlconsensus run --data data/bird.json --backend vllm --model Qwen/Qwen2.5-Coder-7B-Instruct --out-dir outputs/bird/Qwen2.5-Coder-7B-Instruct
```

## Run evaluation: API model

```bash
export OPENAI_API_KEY=...
uv run sqlconsensus run --data data/bird.json --backend openai --model gpt-6-luna --out-dir outputs/bird/gpt-6-luna
```

## Run evaluation: EHRSQL

```bash
uv run sqlconsensus run --data data/ehrsql_eicu.json --backend vllm --model microsoft/phi-4 --ehrsql-repo EHRSQL --out-dir outputs/ehrsql_eicu/phi-4
```

## Run all experiments

```bash
EHRSQL=EHRSQL bash scripts/run_paper.sh
```
