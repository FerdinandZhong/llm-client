#!/bin/bash
set -e

prestart_worker_first_driver=0 ray start --head --disable-usage-stats --include-dashboard true --dashboard-host 0.0.0.0