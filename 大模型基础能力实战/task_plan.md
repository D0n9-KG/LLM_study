# Remote 4090D Fine-Tuning Setup Plan

## Goal
Set up the rented 4090D server for the current Mengzi T5 fine-tuning task, upload the notebook/project assets, and verify the remote environment can run the training workload.

## Phases
- [completed] Inspect local project contents and available SSH tooling
- [completed] Connect to remote server and inspect OS, Python, disk, CUDA, and working directories
- [completed] Create remote project directory and transfer notebook, dataset, and local model files
- [completed] Set up Python environment and install required packages on remote server
- [completed] Verify model loading and dataset access on remote server
- [pending] Summarize remote paths, commands, and any remaining risks

## Decisions
- Use the existing local project root as the upload source.
- Prefer a self-contained remote project directory under `/root`.

## Errors Encountered
- SFTP transfer over the current SSH route is slow and unstable for large files; switched to a hybrid approach: upload small files directly and download the large model file from the remote server with `wget -c`.
