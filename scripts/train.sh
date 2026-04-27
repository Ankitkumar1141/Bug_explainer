#!/usr/bin/env bash
set -euo pipefail
python -m bug_explainer.train --config configs/train_config.yaml
