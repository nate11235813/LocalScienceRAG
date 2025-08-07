#!/usr/bin/env python
"""Switch to a working model while keeping GPT-OSS config for later."""

import yaml
from pathlib import Path

# Read current config
config_path = Path("config/settings.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Save GPT-OSS config as backup
config_backup = config.copy()
with open("config/settings_gpt_oss_backup.yaml", 'w') as f:
    yaml.dump(config_backup, f)
    print("✅ Backed up GPT-OSS config to config/settings_gpt_oss_backup.yaml")

# Update to a working model
config['model']['id'] = "microsoft/phi-2"  # 2.7B model that works reliably
config['model']['dtype'] = "float32"
config['model']['device'] = "cpu"  # Use CPU for compatibility
config['model']['max_new_tokens'] = 200

# Save updated config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Updated config to use {config['model']['id']}")
    print("   This is a smaller but working model for testing")
    print("\n📝 To restore GPT-OSS later:")
    print("   cp config/settings_gpt_oss_backup.yaml config/settings.yaml")
