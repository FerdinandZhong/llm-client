#!/bin/bash
set -e

conda activate llm_cultural_values_env
ray start --head --disable-usage-stats --block --include-dashboard true --dashboard-host 0.0.0.0