# Run with Rscript --vanilla; renv's locked project library is selected explicitly.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
project <- normalizePath(file.path(dirname(script), "../.."))
libs <- list.dirs(file.path(project,"renv/library"), recursive=TRUE, full.names=TRUE)
libs <- libs[file.exists(file.path(libs, "rENA", "DESCRIPTION"))]
if (length(libs) != 1) stop("Restore renv.lock before running ENA")
.libPaths(c(libs, .libPaths()))
suppressPackageStartupMessages(library(rENA))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
input <- if (length(args)>=1 && !grepl("[.]yaml$",args[1])) args[1] else "data/ena/ena_input.csv"
back <- if (length(args)>=2) as.integer(args[2]) else 4L
affect <- if (length(args)>=3) as.numeric(args[3]) else .5
if (length(args)>=1 && grepl("[.]yaml$",args[1])) {
  load_config <- function(path) {
    current <- yaml::read_yaml(path)
    if (!is.null(current$extends)) {
      parent <- load_config(file.path(dirname(path),current$extends))
      current$extends <- NULL
      current <- modifyList(parent,current)
    }
    current
  }
  config <- load_config(args[1])
  if (length(args)<2) back <- as.integer(config$ena$window_size_back)
  if (length(args)<3) affect <- as.numeric(config$ena$affect_threshold)
}
if (!back %in% c(2L,4L,6L)) stop("Unsupported stanza window")
if (!affect %in% c(.33,.5,.67)) stop("Unsupported affect threshold")
set.seed(42)
d <- fread(input)
codes <- c("incorrect","hint_used","bottom_hint_used","repeated_attempt","slow_response",
           "frustrated","confused","bored","concentrating")
stopifnot(all(unlist(d[,..codes]) %in% 0:1))
stopifnot(d[,uniqueN(group),by=student_id][,all(V1==1)])
if (any(d[,uniqueN(student_id),by=group]$V1<2)) stop("Too few independent units")
setorder(d,student_id,event_order)
version <- paste0("ena-w",back,"-a",affect,"-",substr(digest::digest(file=input,algo="sha256"),1,12))
out <- file.path(Sys.getenv("ZHIJI_ENA_OUTPUT_ROOT", "artifacts/ena"),version)
if (file.exists(file.path(out,"manifest.json"))) stop("Analysis already frozen")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
message("Accumulating ", nrow(d), " stanzas with rENA ",packageVersion("rENA"))
a <- ena.accumulate.data(units=d[,.(student_id,group)],
    conversation=d[,.(student_id,conversation_id)],metadata=d[,.(student_id,group)],
    codes=d[,..codes],model="EndPoint",weight.by="binary",window.size.back=back)
s <- ena.make.set(a,dimensions=2)
saveRDS(s,file.path(out,"ena_set.rds"))
weights <- as.data.table(s$line.weights)
edge_names <- names(weights)[vapply(weights,function(x) inherits(x,"ena.co.occurrence"),logical(1))]
if (length(edge_names)!=choose(length(codes),2)) stop("Unexpected rENA edge contract")
nodes <- as.data.table(s$rotation$nodes)
node_json <- data.frame(id=as.character(nodes$code),label=as.character(nodes$code),
                        x=as.numeric(nodes[[2]]),y=as.numeric(nodes[[3]]))
if (any(!is.finite(as.matrix(node_json[,c("x","y")])))) stop("Invalid projection")
points <- as.data.table(s$points)
fwrite(points,file.path(out,"unit_points.csv"))
fwrite(weights,file.path(out,"unit_line_weights.csv"))
fwrite(nodes,file.path(out,"node_positions.csv"))
wm <- as.matrix(weights[,..edge_names])
if (any(!is.finite(wm))) stop("Degenerate unit networks: investigate zero-norm rows")
groups <- as.character(weights$group)
stable <- wm[groups=="stable",,drop=FALSE]
difficulty <- wm[groups=="difficulty",,drop=FALSE]
delta <- colMeans(difficulty)-colMeans(stable)
boot <- replicate(1000L,colMeans(difficulty[sample(nrow(difficulty),replace=TRUE),,drop=FALSE])-
                          colMeans(stable[sample(nrow(stable),replace=TRUE),,drop=FALSE]))
ci <- t(apply(boot,1,quantile,probs=c(.025,.975)))
observed <- sum(delta^2)
null <- replicate(1000L,{
  perm <- sample(groups)
  sum((colMeans(wm[perm=="difficulty",,drop=FALSE])-colMeans(wm[perm=="stable",,drop=FALSE]))^2)
})
edges <- strsplit(edge_names," & ",fixed=TRUE)
edge_table <- function(values) data.frame(source=vapply(edges,`[`,character(1),1),
                                         target=vapply(edges,`[`,character(1),2),weight=as.numeric(values))
write <- function(x,name) write_json(x,file.path(out,name),auto_unbox=TRUE,pretty=TRUE,digits=15,na="null")
for (group in c("stable","difficulty")) {
  values <- colMeans(wm[groups==group,,drop=FALSE])
  network <- list(group=group,unit_count=sum(groups==group),nodes=node_json,edges=edge_table(values),
                  window_size_back=back,affect_threshold=affect,analysis_version=version)
  write(network,paste0(group,"_network.json"))
}
diff_edges <- cbind(edge_table(delta),ci_low=ci[,1],ci_high=ci[,2])
write(list(left="difficulty",right="stable",nodes=node_json,edges=diff_edges,
           unit_counts=as.list(table(groups)),window_size_back=back,affect_threshold=affect,
           analysis_version=version,permutation_p=(1+sum(null>=observed))/1001),"difference_network.json")
fwrite(diff_edges,file.path(out,"difference_network.csv"))
fwrite(data.table(group=c("stable","difficulty"),rbind(colMeans(stable),colMeans(difficulty))),file.path(out,"group_means.csv"))
eigen <- as.numeric(s$rotation$eigenvalues)
write(list(eigenvalues=eigen,proportions=eigen/sum(eigen)),"variance.json")
write(node_json,"node_positions.json")
# SVG/HTML are exported by the dependency-free Python artifact exporter.
writeLines('<!doctype html><meta charset="utf-8"><title>ENA networks</title><h1>Student-unit ENA</h1><p>Co-occurrence does not imply causation. Coordinates and edge scales are shared.</p><img src="plot.svg" alt="Stable, difficulty and difference networks">',file.path(out,"plot.html"))
write(list(artifact_type="ena",analysis_version=version,input_sha256=digest::digest(file=input,algo="sha256"),
           rena_version=as.character(packageVersion("rENA")),window_size_back=back,affect_threshold=affect,
           model="EndPoint",weight_by="binary",seed=42,bootstrap_repeats=1000,permutation_repeats=1000,
           edge_intervals="pointwise; exploratory, not multiplicity-adjusted",unit_count=nrow(wm),
           created_at=format(Sys.time(),tz="UTC",usetz=TRUE)),"manifest.json")
message("Frozen ENA analysis: ",version)
