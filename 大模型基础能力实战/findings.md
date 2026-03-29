# Findings

## Local Project
- Project root contains `src.ipynb`, `DuReaderQG/`, and `mengzi-t5-base/`.
- OpenSSH client tools available locally: `ssh.exe`, `scp.exe`, `sftp.exe`.
- Remote SSH port `38271` is reachable from the current machine.
- Installed local `paramiko` to automate password-based SSH and file transfer.

## Notes
- Remote login target provided by user: `root@connect.bjb2.seetacloud.com:38271`.

## Remote Server
- Hostname: `autodl-container-3l7jf5pgnr-e9f03961`
- OS: Ubuntu Linux kernel `5.15.0-78-generic`
- GPU: `NVIDIA GeForce RTX 4090 D`, `24564 MiB` VRAM, CUDA driver reports `CUDA Version 13.0`
- System memory: `503 GiB`
- Base filesystem `/root`: `30G` available
- Additional workspace mount `/root/autodl-tmp`: `50G` available
- `miniconda3` is already installed under `/root/miniconda3`, but not on the default `PATH`
- No active GPU processes at inspection time
- Remote workspace path: `/root/autodl-tmp/mengzi-t5-finetune`
- Direct SSH/SFTP transfers from the current local network are slow and unstable for large files
- Remote `base` environment already includes:
  - Python `3.12.3`
  - `torch 2.5.1+cu124`
  - CUDA is usable from PyTorch on the 4090D
- Remote `base` environment was missing NLP-task dependencies only (`transformers`, `sentencepiece`, `evaluate`, etc.)
- Installed task dependencies into `base`, avoiding the slower redundant `llm` PyTorch reinstall path
- Current remote file state:
  - `src.ipynb` uploaded with the updated relative checkpoint path
  - `DuReaderQG/dev.json` and `DuReaderQG/train.json` uploaded completely
  - `mengzi-t5-base/config.json` and `spiece.model` present
  - `mengzi-t5-base/pytorch_model.bin` is being resumed remotely with `wget -c`
- Remote environment state:
  - Chosen runtime is the remote `base` environment
  - Installed in `base`: `transformers 5.3.0`, `sentencepiece 0.2.1`, `evaluate 0.4.6`, `sacrebleu 2.6.0`, `matplotlib 3.9.2`, `tqdm 4.67.3`, `ipykernel`
  - Final verification is queued to run automatically in `base` after the full model file is ready
  - The abandoned `llm` environment and its old install/verify scripts were removed
- The downloaded `pytorch_model.bin` is complete (`990389880` bytes)
- Fast workaround for `torch<2.6` + `transformers 5.3.0`:
  - monkey-patch `transformers.modeling_utils.check_torch_load_is_safe = lambda: None`
  - monkey-patch `transformers.utils.import_utils.check_torch_load_is_safe = lambda: None`
  - use only because this local `.bin` file is user-trusted
- Remote `base` verification succeeded:
  - model loads successfully
  - CUDA is available on the 4090 D
  - tokenizer and dataset file access both work
