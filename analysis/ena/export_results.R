# Export and validate the R-produced artifacts through the shared Python contract.
args <- commandArgs(trailingOnly=TRUE)
config <- if (length(args)) args[1] else "configs/full.yaml"
python <- file.path(".venv", "bin", "python")
if (!file.exists(python)) stop("Run uv sync first")
status <- system2(python, c("-m", "zhiji.cli", "ena", "validate-artifacts", "--config", shQuote(config)))
if (status != 0) quit(status=status)
